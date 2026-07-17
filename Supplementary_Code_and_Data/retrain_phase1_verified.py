import os, sys, json, csv, time, glob, re, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path

PROJECT = os.path.dirname(os.path.abspath(__file__))
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join(PROJECT, f"outputs/phase1_retrain_{TIMESTAMP}")
os.makedirs(LOG_DIR, exist_ok=True)

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

def discover_data():
    print("\n[STEP 1] Discovering available data...")
    data = {"images": None, "annotations": {}}
    
    img_candidates = [
        "C:/Users/Admin/Desktop/123/data/coco/val2017",
        "C:/Users/Admin/Desktop/123/DualPath_VLM_GPUServer_Rebuttal/images/val2014",
        "C:/Users/Admin/Desktop/123/data/coco/images/val2014"
    ]
    for d in img_candidates:
        if os.path.isdir(d):
            imgs = [f for f in os.listdir(d) if f.endswith(".jpg")]
            if len(imgs) > 100:
                data["images"] = d
                print(f"  Images: {d} ({len(imgs)} files)")
                break
                
    anno_candidates = [
        "C:/Users/Admin/Desktop/123/data/coco/annotations/captions_val2017.json",
        "C:/Users/Admin/Desktop/123/data/coco/annotations/captions_val2014.json"
    ]
    for anno_path in anno_candidates:
        if os.path.exists(anno_path):
            try:
                with open(anno_path, "r") as f: coco_data = json.load(f)
                for anno in coco_data.get("annotations", []):
                    img_id = anno["image_id"]
                    data["annotations"].setdefault(f"{int(img_id):012d}.jpg", []).append(anno["caption"])
                    data["annotations"].setdefault(f"COCO_val2014_{int(img_id):012d}.jpg", []).append(anno["caption"])
                print(f"  ✓ Loaded ground-truth captions from: {anno_path}")
                break
            except Exception as e:
                print(f"  [WARN] Found annotations but failed to parse: {e}")
                
    if not data["annotations"]:
        print("  [WARN] No COCO annotations found. Will use self-supervised generation fallback for targets.")
        
    return data

def load_model():
    print("\n[STEP 2] Loading model with FRESH Pathway F weights...")
    from model.dualpath_vlm import DualPathVLM
    model = DualPathVLM(model_id="Qwen/Qwen2.5-VL-7B-Instruct", device=DEVICE, use_4bit=False, gradient_checkpointing=True)
    
    from model.pathway_f import FabricationSuppressor
    from model.pathway_o import OmissionCorrector
    from model.gating_mlp import GatingMLP
    
    model.pathway_f = FabricationSuppressor(model.hidden_dim, getattr(model.backbone.config, "vocab_size", 152064)).to(DEVICE).to(model.target_dtype)
    model.pathway_o = OmissionCorrector(model.hidden_dim).to(DEVICE).to(model.target_dtype)
    model.gating = GatingMLP(model.hidden_dim).to(DEVICE).to(model.target_dtype)
    
    # === REVIEWER FIX: CAST RAW_LAMBDA TO FP32 TO PREVENT BFLOAT16 UNDERFLOW ===
    model.pathway_f.raw_lambda.data = model.pathway_f.raw_lambda.data.to(torch.float32)
    # ==========================================================================
    
    for name, param in model.named_parameters():
        param.requires_grad = False
    for param in model.pathway_f.parameters():
        param.requires_grad = True
    
    print("  ✓ Pathway F: Unfrozen and ready for Phase 1 training.")
    return model

def create_training_data(model, data_info, processor, n_samples=1000):
    from PIL import Image
    img_dir = data_info["images"]
    if not img_dir: sys.exit("\n[STEP 3] ✗ No image directory found! Cannot train.")
    
    print(f"\n[STEP 3] Constructing {n_samples} verified training samples with ground-truth targets...")
    all_imgs = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    random.seed(42)
    random.shuffle(all_imgs)
    selected = all_imgs[:n_samples]
    
    samples = []
    fallback_count = 0
    for i, img_name in enumerate(selected):
        img_path = os.path.join(img_dir, img_name)
        try:
            img = Image.open(img_path).convert("RGB").resize((448, 448))
            prompt = "Describe this image in detail."
            
            target_caption = None
            if img_name in data_info["annotations"]:
                target_caption = data_info["annotations"][img_name][0]
            
            if not target_caption:
                fallback_count += 1
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
            if (i+1) % 100 == 0: print(f"  Processed {i+1}/{n_samples} images...")
        except Exception as e:
            print(f"CRASH on {img_name}: {e}")
            continue
            
    split = int(len(samples) * 0.8)
    print(f"  ✓ Train: {len(samples[:split])}, Val: {len(samples[split:])}")
    print(f"  ✓ Samples using generated fallback captions: {fallback_count}/{len(samples)}")
    return samples[:split], samples[split:]

def train_phase1(model, train_samples, val_samples, n_epochs=3):
    log_file = os.path.join(LOG_DIR, "training_log.csv")
    signal_file = os.path.join(LOG_DIR, "fsignal_log.csv")
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4, weight_decay=0.01)
    
    print(f"\n[STEP 4-6] Training Phase 1 for {n_epochs} epochs...")
    with open(log_file, "w", newline="") as f: csv.writer(f).writerow(["timestamp", "epoch", "step", "loss", "type"])
    with open(signal_file, "w", newline="") as f: csv.writer(f).writerow(["epoch", "f_signal_mean", "raw_lambda"])
    
    grad_accum_steps = 4
    for epoch in range(n_epochs):
        model.train()
        epoch_losses, epoch_signals = [], []
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
                
                # ---> REVIEWER DIAGNOSTIC: Print .grad BEFORE zero_grad() <---
                if step % 25 == 0:
                    grad_val = model.pathway_f.raw_lambda.grad
                    if grad_val is not None:
                        print(f"    [Diagnostic] gate_g sample: {outputs['gate_g'].flatten()[:5].tolist()}")
                        print(f"    [Diagnostic] raw_lambda.grad (BEFORE zero_grad): tensor({grad_val.item():.12f}, dtype={grad_val.dtype})")
                    else:
                        print(f"    [Diagnostic] raw_lambda.grad (BEFORE zero_grad): None")
                    
                    raw_lam = model.pathway_f.raw_lambda
                    print(f"    [Diagnostic] raw_lambda.data: {raw_lam.item():.10f}, dtype={raw_lam.dtype}")
                
                if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_samples):
                    torch.nn.utils.clip_grad_norm_(model.pathway_f.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                loss_val = loss.item()
                epoch_losses.append(loss_val)
                
                with torch.no_grad():
                    sig_val = model.monitor.get_fabrication_signal().mean().item()
                    epoch_signals.append(sig_val)
                
                if step % 25 == 0:
                    with open(log_file, "a", newline="") as f: csv.writer(f).writerow([datetime.now().isoformat(), epoch+1, step, f"{loss_val:.6f}", "train"])
                    print(f"    Step {step:4d}/{len(train_samples)} | Loss: {loss_val:.4f} | f_sig: {sig_val:.4f}")
        
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        sig_mean = sum(epoch_signals) / len(epoch_signals) if epoch_signals else 0
        raw_lam = model.pathway_f.raw_lambda.item() if hasattr(model.pathway_f, "raw_lambda") else 0.0
        
        with open(signal_file, "a", newline="") as f: csv.writer(f).writerow([epoch+1, f"{sig_mean:.6f}", f"{raw_lam:.4f}"])
        print(f"  Epoch {epoch+1} TRAIN: avg_loss={avg_loss:.4f}, f_signal={sig_mean:.4f}")
        print(f"  Pathway F: raw_lambda={raw_lam:.4f}")
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for sample in val_samples:
                inputs = {"input_ids": sample["input_ids"].unsqueeze(0).to(DEVICE), "labels": sample["labels"].unsqueeze(0).to(DEVICE)}
                if sample["pixel_values"] is not None:
                    inputs["pixel_values"] = sample["pixel_values"].unsqueeze(0).to(DEVICE)
                    inputs["image_grid_thw"] = sample["image_grid_thw"].unsqueeze(0).to(DEVICE)
                out = model(**inputs)
                if out["loss"] is not None: val_losses.append(out["loss"].item())
        
        if val_losses:
            avg_val = sum(val_losses) / len(val_losses)
            with open(log_file, "a", newline="") as f: csv.writer(f).writerow([datetime.now().isoformat(), epoch+1, "VAL", f"{avg_val:.6f}", "val"])
            print(f"  Epoch {epoch+1} VAL:   avg_loss={avg_val:.4f}")
            
    ckpt_path = os.path.join(LOG_DIR, "pathway_f_retrained.pt")
    torch.save({"pathway_f": model.pathway_f.state_dict(), "pathway_o": model.pathway_o.state_dict(), "gating": model.gating.state_dict()}, ckpt_path)
    print(f"\n  ✓ Checkpoint saved: {ckpt_path}")
    return model

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
        except Exception: pass

def main():
    fix_attention_monitor()
    data_info = discover_data()
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    model = load_model()
    train_samples, val_samples = create_training_data(model, data_info, processor, n_samples=1000)
    model = train_phase1(model, train_samples, val_samples, n_epochs=3)
    run_diagnostic(model, processor, data_info)

if __name__ == "__main__":
    main()
