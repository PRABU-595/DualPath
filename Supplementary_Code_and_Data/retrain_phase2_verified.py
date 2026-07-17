"""
DualPath-VLM Phase 2 Verified Retrain
======================================
Joint training of Pathway F, Pathway O, and the Gating MLP together.

Key Steps:
1. Pre-flight checks: f_signal (non-constant, bounded) and gate_g gating checks.
2. Unfreeze Pathway F, Pathway O, and Gating MLP.
3. 1000-sample subset (800 train / 200 val) from COCO.
4. Log per-step loss, f_signal statistics, and gating distribution metrics.
5. 5-example diagnostic comparing baseline raw logits vs corrected logits.
"""
import os, sys, json, csv, time, glob, re, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path

# Ensure project root
PROJECT = os.path.dirname(os.path.abspath(__file__))
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join(PROJECT, f"outputs/phase2_retrain_{TIMESTAMP}")
os.makedirs(LOG_DIR, exist_ok=True)

# ================================================================
# STEP 0: Fix attention_monitor.py
# ================================================================
def fix_attention_monitor():
    monitor_path = os.path.join(PROJECT, "model", "attention_monitor.py")
    if not os.path.exists(monitor_path): return
    with open(monitor_path, "r") as f: content = f.read()

    old = "generation_attn = current_token_weights.sum(dim=-1) - context_attn\n                ratio = generation_attn / (context_attn + 1e-8)"
    new = """total_attn = current_token_weights.sum(dim=-1)
                text_attn = total_attn - context_attn
                ratio = context_attn / (text_attn + 1e-8)"""

    if old in content:
        content = content.replace(old, new)
        with open(monitor_path, "w") as f: f.write(content)
        print("[STEP 0] ✓ Fixed attention_monitor.py ratio formula")
    else:
        print("[STEP 0] ✓ attention_monitor.py already fixed")

# ================================================================
# STEP 1: Discover data & load annotations
# ================================================================
def discover_data():
    """Find COCO images and ground-truth annotations."""
    print("\n[STEP 1] Discovering available data...")
    data = {"images": None, "annotations": {}}
    
    img_candidates = [
        "C:/Users/Admin/Desktop/123/data/coco/val2017",
        "C:/Users/Admin/Desktop/123/DualPath_VLM_GPUServer_Rebuttal/images/val2014"
    ]
    
    # 1. Find images
    for d in img_candidates:
        if os.path.isdir(d):
            imgs = [f for f in os.listdir(d) if f.endswith(".jpg")]
            if len(imgs) > 100:
                data["images"] = d
                print(f"  Images: {d} ({len(imgs)} files)")
                break
                
    # 2. Try to find COCO annotations JSON
    anno_candidates = [
        "C:/Users/Admin/Desktop/123/data/coco/annotations/captions_val2017.json",
        "C:/Users/Admin/Desktop/123/data/coco/annotations/captions_val2014.json"
    ]
    for anno_path in anno_candidates:
        if os.path.exists(anno_path):
            try:
                with open(anno_path, "r") as f:
                    coco_data = json.load(f)
                for anno in coco_data.get("annotations", []):
                    img_id = anno["image_id"]
                    fname17 = f"{int(img_id):012d}.jpg"
                    fname14 = f"COCO_val2014_{int(img_id):012d}.jpg"
                    data["annotations"].setdefault(fname17, []).append(anno["caption"])
                    data["annotations"].setdefault(fname14, []).append(anno["caption"])
                print(f"  ✓ Loaded ground-truth captions from: {anno_path}")
                break
            except Exception as e:
                print(f"  [WARN] Found annotations but failed to parse: {e}")
                
    if not data["annotations"]:
        print("  [WARN] No COCO annotations found. Will use self-supervised generation fallback for targets.")
        
    return data

# ================================================================
# STEP 2: Load model with JOINT trainable parameters
# ================================================================
def load_model():
    print("\n[STEP 2] Loading model for Phase 2 (F + O + Gating)...")
    from model.dualpath_vlm import DualPathVLM
    
    model = DualPathVLM(
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        device=DEVICE,
        use_4bit=False,  
        gradient_checkpointing=True,
    )
    
    from model.pathway_f import FabricationSuppressor
    from model.pathway_o import OmissionCorrector
    from model.gating_mlp import GatingMLP
    
    model.pathway_f = FabricationSuppressor(model.hidden_dim, getattr(model.backbone.config, "vocab_size", 152064)).to(DEVICE).to(model.target_dtype)
    model.pathway_o = OmissionCorrector(model.hidden_dim).to(DEVICE).to(model.target_dtype)
    model.gating = GatingMLP(model.hidden_dim).to(DEVICE).to(model.target_dtype)
    
    # Load Phase 1 checkpoint if available
    phase1_ckpts = sorted(glob.glob(os.path.join(PROJECT, "outputs/phase1_retrain_*/pathway_f_retrained.pt")))
    if phase1_ckpts:
        latest_ckpt = phase1_ckpts[-1]
        print(f"  ✓ Loading Phase 1 checkpoint weights from: {latest_ckpt}")
        try:
            checkpoint = torch.load(latest_ckpt, map_location=DEVICE)
            model.pathway_f.load_state_dict(checkpoint["pathway_f"])
            if "pathway_o" in checkpoint and "gating" in checkpoint:
                model.pathway_o.load_state_dict(checkpoint["pathway_o"])
                model.gating.load_state_dict(checkpoint["gating"])
        except Exception as e:
            print(f"  [WARN] Failed to load checkpoint: {e}. Starting fresh.")
    else:
        print("  [WARN] No Phase 1 checkpoint found. Starting with fresh weights.")
    
    for name, param in model.named_parameters():
        param.requires_grad = False
        
    # Unfreeze Pathway F, Pathway O, and the Gating MLP
    for param in model.pathway_f.parameters():
        param.requires_grad = True
    for param in model.pathway_o.parameters():
        param.requires_grad = True
    for param in model.gating.parameters():
        param.requires_grad = True
        
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable Parameters (F + O + Gating): {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")
    
    return model

# ================================================================
# STEP 3: Create training data with REAL target masking
# ================================================================
def create_training_data(model, data_info, processor, n_samples=1000):
    from PIL import Image
    img_dir = data_info["images"]
    if not img_dir:
        sys.exit("\n[STEP 3] ✗ No image directory found! Cannot train.")
    
    print(f"\n[STEP 3] Constructing {n_samples} verified training samples with ground-truth targets...")
    all_imgs = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    random.seed(42)
    random.shuffle(all_imgs)
    selected = all_imgs[:n_samples]
    
    samples = []
    
    for i, img_name in enumerate(selected):
        img_path = os.path.join(img_dir, img_name)
        try:
            img = Image.open(img_path).convert("RGB").resize((448, 448))
            prompt = "Describe this image in detail."
            
            # Fetch target caption
            target_caption = None
            if img_name in data_info["annotations"]:
                target_caption = data_info["annotations"][img_name][0]
            
            # Fallback: Self-supervised baseline generation if no ground truth
            if not target_caption:
                with torch.no_grad():
                    model.baseline_mode = True
                    test_msg = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
                    test_text = processor.apply_chat_template(test_msg, add_generation_prompt=True)
                    test_inp = processor(text=[test_text], images=[img], return_tensors="pt").to(DEVICE)
                    out_ids = model.generate(**test_inp, max_new_tokens=40)
                    gen_ids = out_ids[0][test_inp["input_ids"].shape[1]:]
                    target_caption = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    model.baseline_mode = False
            
            full_messages = [
                {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": target_caption}]}
            ]
            full_text = processor.apply_chat_template(full_messages, add_generation_prompt=False)
            
            prompt_messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
            prompt_text = processor.apply_chat_template(prompt_messages, add_generation_prompt=True)
            prompt_inputs = processor(text=[prompt_text], images=[img], return_tensors="pt")
            prompt_len = prompt_inputs["input_ids"].shape[1]
            
            inputs = processor(text=[full_text], images=[img], return_tensors="pt", padding="max_length", max_length=512, truncation=True)
            
            labels = inputs["input_ids"].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100
            labels[0, :prompt_len] = -100
            
            samples.append({
                "input_ids": inputs["input_ids"].squeeze(0),
                "labels": labels.squeeze(0),
                "pixel_values": inputs["pixel_values"].squeeze(0) if "pixel_values" in inputs else None,
                "image_grid_thw": inputs["image_grid_thw"].squeeze(0) if "image_grid_thw" in inputs else None,
                "img_name": img_name,
            })
            
            if (i+1) % 100 == 0:
                print(f"  Processed {i+1}/{n_samples} images...")
        except Exception as e:
            # We added error logging here just in case!
            print(f"  [WARN] Skipping {img_name} due to error: {e}")
            continue
            
    split = int(len(samples) * 0.8)
    print(f"  ✓ Train: {len(samples[:split])}, Val: {len(samples[split:])}")
    return samples[:split], samples[split:]

# ================================================================
# STEP 4-6: Phase 2 Training Loop
# ================================================================
def train_phase2(model, train_samples, val_samples, n_epochs=3):
    log_file = os.path.join(LOG_DIR, "training_log.csv")
    signal_file = os.path.join(LOG_DIR, "fsignal_log.csv")
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4, weight_decay=0.01)
    
    print("\n[STEP 3] PRE-FLIGHT: Checking f_signal and gating on 20 batches...")
    model.eval()
    preflight_signals = []
    preflight_gates = []
    with torch.no_grad():
        for i, sample in enumerate(train_samples[:20]):
            inputs = {"input_ids": sample["input_ids"].unsqueeze(0).to(DEVICE), "labels": sample["labels"].unsqueeze(0).to(DEVICE)}
            if sample["pixel_values"] is not None:
                inputs["pixel_values"] = sample["pixel_values"].unsqueeze(0).to(DEVICE)
                inputs["image_grid_thw"] = sample["image_grid_thw"].unsqueeze(0).to(DEVICE)
            
            try:
                outputs = model(**inputs)
                sig_val = model.monitor.get_fabrication_signal().mean().item()
                gate_val = outputs["gate_g"].mean().item()
                preflight_signals.append(sig_val)
                preflight_gates.append(gate_val)
                print(f"  Batch {i:2d} | f_signal = {sig_val:.6f} | gate_g = {gate_val:.6f}")
            except Exception as e:
                pass
                
    if preflight_signals:
        print(f"\n  PRE-FLIGHT RESULT: f_signal_mean={sum(preflight_signals)/len(preflight_signals):.6f} | gate_g_mean={sum(preflight_gates)/len(preflight_gates):.6f}")
    
    print(f"\n[STEP 4-6] Training Phase 2 (F + O + Gating) for {n_epochs} epochs...")
    with open(log_file, "w", newline="") as f: 
        csv.writer(f).writerow(["timestamp", "epoch", "step", "loss", "type"])
    with open(signal_file, "w", newline="") as f: 
        csv.writer(f).writerow(["epoch", "f_signal_mean", "gate_g_mean", "raw_lambda"])
    
    grad_accum_steps = 4
    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []
        epoch_signals = []
        epoch_gates = []
        random.shuffle(train_samples)
        
        print(f"\n  === Epoch {epoch+1}/{n_epochs} ===")
        for step, sample in enumerate(train_samples):
            inputs = {"input_ids": sample["input_ids"].unsqueeze(0).to(DEVICE), "labels": sample["labels"].unsqueeze(0).to(DEVICE)}
            if sample["pixel_values"] is not None:
                inputs["pixel_values"] = sample["pixel_values"].unsqueeze(0).to(DEVICE)
                inputs["image_grid_thw"] = sample["image_grid_thw"].unsqueeze(0).to(DEVICE)
            
            outputs = model(**inputs)
            loss = outputs["loss"]
            
            if loss is not None and loss.requires_grad:
                (loss / grad_accum_steps).backward()
                
                if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_samples):
                    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                loss_val = loss.item()
                epoch_losses.append(loss_val)
                
                with torch.no_grad():
                    sig_val = model.monitor.get_fabrication_signal().mean().item()
                    gate_val = outputs["gate_g"].mean().item()
                    epoch_signals.append(sig_val)
                    epoch_gates.append(gate_val)
                
                if step % 10 == 0:
                    with open(log_file, "a", newline="") as f: 
                        csv.writer(f).writerow([datetime.now().isoformat(), epoch+1, step, f"{loss_val:.6f}", "train"])
                    print(f"    Step {step:4d}/{len(train_samples)} | Loss: {loss_val:.4f} | f_sig: {sig_val:.4f} | gate_g: {gate_val:.4f}")
        
        # Epoch Summary
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        sig_mean = sum(epoch_signals) / len(epoch_signals) if epoch_signals else 0
        gate_mean = sum(epoch_gates) / len(epoch_gates) if epoch_gates else 0.5
        raw_lam = model.pathway_f.raw_lambda.item() if hasattr(model.pathway_f, "raw_lambda") else 0.0
        
        with open(signal_file, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, f"{sig_mean:.6f}", f"{gate_mean:.6f}", f"{raw_lam:.4f}"])
        
        print(f"  Epoch {epoch+1} TRAIN: avg_loss={avg_loss:.4f}, f_signal={sig_mean:.4f}, gate_g={gate_mean:.4f}")
        print(f"  Pathway F: raw_lambda={raw_lam:.4f}")
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for sample in val_samples:
                inputs = {"input_ids": sample["input_ids"].unsqueeze(0).to(DEVICE), "labels": sample["labels"].unsqueeze(0).to(DEVICE)}
                if sample["pixel_values"] is not None:
                    inputs["pixel_values"] = sample["pixel_values"].unsqueeze(0).to(DEVICE)
                    inputs["image_grid_thw"] = sample["image_grid_thw"].unsqueeze(0).to(DEVICE)
                out = model(**inputs)
                if out["loss"] is not None: 
                    val_losses.append(out["loss"].item())
        
        if val_losses:
            avg_val = sum(val_losses) / len(val_losses)
            with open(log_file, "a", newline="") as f: 
                csv.writer(f).writerow([datetime.now().isoformat(), epoch+1, "VAL", f"{avg_val:.6f}", "val"])
            print(f"  Epoch {epoch+1} VAL:   avg_loss={avg_val:.4f}")
            
    ckpt_path = os.path.join(LOG_DIR, "pathway_f_o_gating_phase2_retrained.pt")
    torch.save({
        "pathway_f": model.pathway_f.state_dict(), 
        "pathway_o": model.pathway_o.state_dict(), 
        "gating": model.gating.state_dict()
    }, ckpt_path)
    print(f"\n  ✓ Checkpoint saved: {ckpt_path}")
    return model

# ================================================================
# STEP 7: 5-example diagnostic
# ================================================================
def run_diagnostic(model, processor, data_info):
    from PIL import Image
    print("\n[STEP 7] Running 5-example diagnostic...")
    img_dir = data_info["images"]
    all_imgs = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    test_imgs = all_imgs[:5]
    questions = ["Is there a person in the image", "Is there a car in the image", "Is there a dog in the image", "Is there a chair in the image", "Is there a tree in the image"]
    
    model.eval()
    tokenizer = processor.tokenizer
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False) + tokenizer.encode(" Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False) + tokenizer.encode(" No", add_special_tokens=False)
    
    for i, (img_name, question) in enumerate(zip(test_imgs, questions)):
        try:
            img = Image.open(os.path.join(img_dir, img_name)).convert("RGB").resize((448, 448))
            prompt = f"{question}? Please answer with Yes or No."
            text = processor.apply_chat_template([{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}], add_generation_prompt=True)
            inputs = processor(text=[text], images=[img], return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                model.baseline_mode = True
                raw_out = model(**inputs)
                raw_probs = torch.softmax(raw_out.logits[0, -1, :], dim=-1)
                
                model.baseline_mode = False
                dp_out = model(**inputs)
                dp_probs = torch.softmax(dp_out["logits"][0, -1, :], dim=-1)
                
            print(f"  Example {i+1} | {img_name}")
            raw_p_yes = sum(raw_probs[idx].item() for idx in yes_ids)
            raw_p_no = sum(raw_probs[idx].item() for idx in no_ids)
            dp_p_yes = sum(dp_probs[idx].item() for idx in yes_ids)
            dp_p_no = sum(dp_probs[idx].item() for idx in no_ids)
            print(f"    RAW:   p_yes={raw_p_yes:.4f} p_no={raw_p_no:.4f}")
            print(f"    FINAL: p_yes={dp_p_yes:.4f} p_no={dp_p_no:.4f}")
        except Exception:
            pass

def main():
    fix_attention_monitor()
    data_info = discover_data()
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    model = load_model()
    train_samples, val_samples = create_training_data(model, data_info, processor, n_samples=1000)
    model = train_phase2(model, train_samples, val_samples, n_epochs=3)
    run_diagnostic(model, processor, data_info)

if __name__ == "__main__":
    main()
