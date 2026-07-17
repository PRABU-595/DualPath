"""
DualPath-VLM: Multi-Phase Trainer
Complete training pipeline with real data, proper loss computation,
and checkpoint management.

Usage:
    # Standard launch
    python training/trainer.py --config_path configs/base_config.yaml
    
    # With accelerate (multi-GPU)
    accelerate launch training/trainer.py --config_path configs/base_config.yaml
    
    # Dry run (verify data loading only)
    python training/trainer.py --config_path configs/base_config.yaml --dry_run
"""

import sys
import os

# Ensure project root is importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset
from transformers import get_scheduler, AutoProcessor
from accelerate import Accelerator
from tqdm.auto import tqdm
import yaml
import argparse
import json
import time
from typing import Dict, Any, Optional

from training.dual_contrastive_loss import DualContrastiveLoss
from training.checkpoint_manager import CheckpointManager
from data.dataset_loader import DualPathLocalDataset, collate_fn_v2


class Trainer:
    """
    Multi-Phase Training loop for DualPath-VLM.
    Phase 1: Warmup — train Pathway F and O in isolation
    Phase 2: Joint — train F + O + Gating MLP together
    Phase 3: Fine-tuning — unfreeze top backbone layers + all modules
    """
    
    def __init__(self, 
                 model: nn.Module,
                 processor: AutoProcessor,
                 config: Dict[str, Any]):
        
        self.config = config
        training_cfg = config.get("training", {})
        loss_cfg = config.get("loss", {})
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=training_cfg.get("grad_acc_steps", 4),
            mixed_precision=training_cfg.get("mixed_precision", "bf16"),
            log_with="tensorboard",
            project_dir=config.get("paths", {}).get("log_dir", "./outputs/logs"),
        )
        
        self.model = model
        self.processor = processor
        
        self.loss_fn = DualContrastiveLoss(
            lambda_f=loss_cfg.get("lambda_f", 1.0),
            lambda_o=loss_cfg.get("lambda_o", 1.0),
            lambda_g=loss_cfg.get("lambda_g", 0.1),
        )
        
        self.checkpoint_mgr = CheckpointManager(
            save_dir=config.get("paths", {}).get("output_dir", "./outputs/checkpoints")
        )
        
        # Config values
        self.batch_size = training_cfg.get("batch_size", 4)
        self.grad_clip = training_cfg.get("grad_clip", 1.0)
        self.log_every = config.get("evaluation", {}).get("log_every", 50)
        self.save_every = config.get("evaluation", {}).get("save_every", 500)
        self.eval_every = config.get("evaluation", {}).get("eval_every", 1000)
        
        # Output dirs
        self.output_dir = config.get("paths", {}).get("output_dir", "./outputs/checkpoints")
        self.log_dir = config.get("paths", {}).get("log_dir", "./outputs/logs")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
    
    def setup_data(self, 
                   fab_jsonl: str, 
                   om_jsonl: str, 
                   image_dir: str,
                   val_image_dir: Optional[str] = None):
        """Creates DataLoaders from local .jsonl files."""
        
        data_cfg = self.config.get("data", {})
        max_length = data_cfg.get("max_length", 512)
        image_size = data_cfg.get("image_size", 448)
        
        self.accelerator.print(f"Loading fabrication dataset: {fab_jsonl}")
        dataset_f = DualPathLocalDataset(
            jsonl_path=fab_jsonl,
            image_dir=image_dir,
            processor=self.processor,
            max_length=max_length,
            image_size=image_size,
        )
        
        self.accelerator.print(f"Loading omission dataset: {om_jsonl}")
        dataset_o = DualPathLocalDataset(
            jsonl_path=om_jsonl,
            image_dir=image_dir,
            processor=self.processor,
            max_length=max_length,
            image_size=image_size,
        )
        
        # Joint dataset = interleaved F + O samples
        dataset_joint = ConcatDataset([dataset_f, dataset_o])
        
        self.loader_f = DataLoader(
            dataset_f, batch_size=self.batch_size, shuffle=True, 
            collate_fn=collate_fn_v2, num_workers=0, pin_memory=True,
            drop_last=True,
        )
        self.loader_o = DataLoader(
            dataset_o, batch_size=self.batch_size, shuffle=True,
            collate_fn=collate_fn_v2, num_workers=0, pin_memory=True,
            drop_last=True,
        )
        self.loader_joint = DataLoader(
            dataset_joint, batch_size=self.batch_size, shuffle=True,
            collate_fn=collate_fn_v2, num_workers=0, pin_memory=True,
            drop_last=True,
        )
        
        self.accelerator.print(
            f"Data ready: {len(dataset_f)} F pairs, {len(dataset_o)} O pairs, "
            f"{len(dataset_joint)} joint"
        )

    def setup_unified_data(self,
                           vlm_fab_jsonl: str,
                           vlm_om_jsonl: str,
                           image_dir: str,
                           llm_fab_jsonl: str,
                           llm_om_jsonl: str):
        """
        Unified Mode: Loads both VLM (image) and LLM (text) datasets and merges
        them into interleaved DataLoaders for cross-modal training.
        Each batch is drawn alternately from image and text sources, ensuring
        the model sees both modalities every training step.
        """
        data_cfg = self.config.get("data", {})
        max_length = data_cfg.get("max_length", 512)
        image_size = data_cfg.get("image_size", 448)

        # --- VLM Datasets (image+text) ---
        self.accelerator.print("[UNIFIED] Loading VLM (Vision) datasets...")
        vlm_dataset_f = DualPathLocalDataset(
            jsonl_path=vlm_fab_jsonl,
            image_dir=image_dir,
            processor=self.processor,
            max_length=max_length,
            image_size=image_size,
        )
        vlm_dataset_o = DualPathLocalDataset(
            jsonl_path=vlm_om_jsonl,
            image_dir=image_dir,
            processor=self.processor,
            max_length=max_length,
            image_size=image_size,
        )

        # --- LLM Datasets (text-only) ---
        self.accelerator.print("[UNIFIED] Loading LLM (Text) datasets...")
        llm_dataset_f = DualPathLocalDataset(
            jsonl_path=llm_fab_jsonl,
            image_dir=None,   # No images — pure text mode
            processor=self.processor,
            max_length=max_length,
            image_size=image_size,
        )
        llm_dataset_o = DualPathLocalDataset(
            jsonl_path=llm_om_jsonl,
            image_dir=None,   # No images — pure text mode
            processor=self.processor,
            max_length=max_length,
            image_size=image_size,
        )

        # --- Merge all 4 sources into one unified joint dataset ---
        # The collate_fn_v2 already handles missing pixel_values per-batch.
        # ConcatDataset interleaves samples across modalities automatically.
        dataset_unified = ConcatDataset([vlm_dataset_f, vlm_dataset_o,
                                         llm_dataset_f, llm_dataset_o])

        # Phase 1 loaders (modality-specific for warm-up)
        self.loader_f = DataLoader(
            ConcatDataset([vlm_dataset_f, llm_dataset_f]),
            batch_size=self.batch_size, shuffle=True,
            collate_fn=collate_fn_v2, num_workers=0, pin_memory=True,
            drop_last=True,
        )
        self.loader_o = DataLoader(
            ConcatDataset([vlm_dataset_o, llm_dataset_o]),
            batch_size=self.batch_size, shuffle=True,
            collate_fn=collate_fn_v2, num_workers=0, pin_memory=True,
            drop_last=True,
        )
        # Phase 2 & 3 joint loader (all modalities mixed)
        self.loader_joint = DataLoader(
            dataset_unified, batch_size=self.batch_size, shuffle=True,
            collate_fn=collate_fn_v2, num_workers=0, pin_memory=True,
            drop_last=True,
        )

        n_vlm = len(vlm_dataset_f) + len(vlm_dataset_o)
        n_llm = len(llm_dataset_f) + len(llm_dataset_o)
        self.accelerator.print(
            f"[UNIFIED] Data ready:"
            f"  VLM pairs={n_vlm} | LLM pairs={n_llm} | Total={n_vlm + n_llm}"
        )
    
    def _setup_optimizer(self, phase: int) -> AdamW:
        """Sets up optimizer with phase-specific parameter groups and learning rates."""
        training_cfg = self.config.get("training", {})
        weight_decay = training_cfg.get("weight_decay", 0.01)
        
        if phase == 1:
            # Warmup: Only Pathway F and O parameters
            params = [
                {"params": self.model.pathway_f.parameters(), 
                 "lr": training_cfg.get("lr_warmup", 2e-4)},
                {"params": self.model.pathway_o.parameters(), 
                 "lr": training_cfg.get("lr_warmup", 2e-4)},
            ]
        elif phase == 2:
            # Joint: F, O, and Gating MLP
            params = [
                {"params": self.model.pathway_f.parameters(), 
                 "lr": training_cfg.get("lr_joint", 1e-4)},
                {"params": self.model.pathway_o.parameters(), 
                 "lr": training_cfg.get("lr_joint", 1e-4)},
                {"params": self.model.gating.parameters(), 
                 "lr": training_cfg.get("lr_joint", 1e-4)},
            ]
        else:
            # Fine-tuning: All modules + LoRA backbone
            params = [
                {"params": self.model.pathway_f.parameters(), 
                 "lr": training_cfg.get("lr_finetune", 1e-5)},
                {"params": self.model.pathway_o.parameters(), 
                 "lr": training_cfg.get("lr_finetune", 1e-5)},
                {"params": self.model.gating.parameters(), 
                 "lr": training_cfg.get("lr_finetune", 1e-5)},
                # LoRA parameters in backbone (PEFT makes only LoRA params require_grad)
                {"params": [p for p in self.model.backbone.parameters() if p.requires_grad], 
                 "lr": training_cfg.get("lr_finetune", 1e-5)},
            ]
        
        return AdamW(params, weight_decay=weight_decay)
    
    def _train_epoch(self, dataloader, optimizer, scheduler, epoch: int, 
                     phase_name: str, target_pathway: str = "Joint") -> Dict[str, float]:
        """Runs one training epoch and returns average metrics."""
        self.model.train()
        
        total_loss = 0.0
        total_fab_loss = 0.0
        total_om_loss = 0.0
        total_gate_loss = 0.0
        num_steps = 0
        
        progress_bar = tqdm(
            dataloader, 
            desc=f"Phase {phase_name} / Epoch {epoch + 1}",
            disable=not self.accelerator.is_local_main_process,
        )
        
        for step, batch in enumerate(progress_bar):
            with self.accelerator.accumulate(self.model):
                # Forward pass through DualPath-VLM
                outputs = self.model(
                    input_ids=batch["inputs"]["input_ids"],
                    pixel_values=batch["inputs"]["pixel_values"],
                    image_grid_thw=batch["inputs"]["image_grid_thw"],
                    labels=batch["inputs"]["labels"],
                )
                
                # Get gate values and expand to match loss expectations
                gate_g = outputs.get("gate_g")  # [batch, 1]
                if gate_g is not None:
                    seq_len = batch["inputs"]["input_ids"].shape[1]
                    # Expand gate to [batch, seq_len, 1] for gating loss
                    gate_g_expanded = gate_g.unsqueeze(1).expand(-1, seq_len, -1)
                else:
                    gate_g_expanded = None
                
                # Compute DualContrastive loss
                loss_dict = self.loss_fn(
                    logits=outputs["logits"],
                    good_labels=batch["inputs"]["labels"],
                    fabricated_token_ids=batch.get("fabricated_tokens", []),
                    omitted_token_ids=batch.get("omitted_tokens", []),
                    gate_values=gate_g_expanded,
                    pathway=target_pathway,
                )
                
                loss = loss_dict["loss"]
                
                self.accelerator.backward(loss)
                
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                num_steps += 1
                total_loss += loss.item()
                total_fab_loss += loss_dict["fabrication_loss"].item() if isinstance(loss_dict["fabrication_loss"], torch.Tensor) else 0
                total_om_loss += loss_dict["omission_loss"].item() if isinstance(loss_dict["omission_loss"], torch.Tensor) else 0
                total_gate_loss += loss_dict["gate_regularization"].item() if isinstance(loss_dict["gate_regularization"], torch.Tensor) else 0
                
                # Update progress bar
                if num_steps % self.log_every == 0:
                    avg = total_loss / num_steps
                    progress_bar.set_postfix({
                        "loss": f"{avg:.4f}",
                        "L_F": f"{total_fab_loss / num_steps:.4f}",
                        "L_O": f"{total_om_loss / num_steps:.4f}",
                    })
                
                # Periodic checkpoint
                global_step = epoch * len(dataloader) + step
                if num_steps % self.save_every == 0 and self.accelerator.is_main_process:
                    self.checkpoint_mgr.save_checkpoint(
                        self.accelerator.unwrap_model(self.model),
                        step=global_step,
                        optimizer=optimizer,
                    )
        
        metrics = {
            "avg_loss": total_loss / max(num_steps, 1),
            "avg_fab_loss": total_fab_loss / max(num_steps, 1),
            "avg_om_loss": total_om_loss / max(num_steps, 1),
            "avg_gate_loss": total_gate_loss / max(num_steps, 1),
            "num_steps": num_steps,
        }
        
        self.accelerator.print(
            f"  Epoch {epoch + 1} complete | "
            f"Loss: {metrics['avg_loss']:.4f} | "
            f"L_F: {metrics['avg_fab_loss']:.4f} | "
            f"L_O: {metrics['avg_om_loss']:.4f} | "
            f"L_Gate: {metrics['avg_gate_loss']:.4f} | "
            f"Steps: {metrics['num_steps']}"
        )
        
        return metrics
    
    def train(self):
        """Executes the complete 3-phase training pipeline."""
        latest = self.checkpoint_mgr.find_latest_checkpoint()
        if latest:
            self.accelerator.print(f"[RESUME] Found existing checkpoint: {latest}. Loading weights...")
            self.checkpoint_mgr.load_checkpoint(self.model, latest)
            
        training_cfg = self.config.get("training", {})
        epochs_p1 = training_cfg.get("epochs_phase1", 5)
        epochs_p2 = training_cfg.get("epochs_phase2", 15)
        epochs_p3 = training_cfg.get("epochs_phase3", 5)
        
        all_metrics = []
        
        # ==============================
        # Phase 1: Warmup
        # ==============================
        self.accelerator.print("\n" + "=" * 60)
        self.accelerator.print("PHASE 1: WARMUP (Pathway F and O isolated)")
        self.accelerator.print("=" * 60)
        
        opt_p1 = self._setup_optimizer(phase=1)
        total_steps_p1 = epochs_p1 * (len(self.loader_f) + len(self.loader_o))
        sched_p1 = get_scheduler(
            "constant_with_warmup", optimizer=opt_p1,
            num_warmup_steps=min(500, total_steps_p1 // 10),
            num_training_steps=total_steps_p1,
        )
        
        # Prepare with accelerator
        self.model, opt_p1, self.loader_f, self.loader_o = self.accelerator.prepare(
            self.model, opt_p1, self.loader_f, self.loader_o
        )
        
        for ep in range(epochs_p1):
            self.accelerator.print(f"\n--- Phase 1, Epoch {ep + 1}/{epochs_p1} ---")
            # [SKIP] Pathway F is already fully trained from checkpoint
            self.accelerator.print(f"Skipping Phase 1-F (Fabrication) as it is already complete.")
            m_f = {"avg_loss": 0, "avg_fab_loss": 0, "avg_om_loss": 0, "avg_gate_loss": 0, "num_steps": 0}
            # Train Pathway O on omission data
            m_o = self._train_epoch(
                self.loader_o, opt_p1, sched_p1, ep, "1-O (Omission)", target_pathway="O"
            )
            all_metrics.append({"phase": 1, "epoch": ep, "fab": m_f, "om": m_o})
        
        # ==============================
        # Phase 2: Joint Training
        # ==============================
        self.accelerator.print("\n" + "=" * 60)
        self.accelerator.print("PHASE 2: JOINT TRAINING (F + O + Gating MLP)")
        self.accelerator.print("=" * 60)
        
        opt_p2 = self._setup_optimizer(phase=2)
        total_steps_p2 = epochs_p2 * len(self.loader_joint)
        sched_p2 = get_scheduler(
            "cosine", optimizer=opt_p2,
            num_warmup_steps=min(500, total_steps_p2 // 10),
            num_training_steps=total_steps_p2,
        )
        
        opt_p2, self.loader_joint = self.accelerator.prepare(opt_p2, self.loader_joint)
        
        best_loss = float("inf")
        for ep in range(epochs_p2):
            self.accelerator.print(f"\n--- Phase 2, Epoch {ep + 1}/{epochs_p2} ---")
            m = self._train_epoch(
                self.loader_joint, opt_p2, sched_p2, ep, "2-Joint", target_pathway="Joint"
            )
            all_metrics.append({"phase": 2, "epoch": ep, "joint": m})
            
            # Save best model
            if m["avg_loss"] < best_loss and self.accelerator.is_main_process:
                best_loss = m["avg_loss"]
                self.checkpoint_mgr.save_checkpoint(
                    self.accelerator.unwrap_model(self.model),
                    step=ep, is_best=True,
                )
        
        # ==============================
        # Phase 3: Fine-tuning
        # ==============================
        self.accelerator.print("\n" + "=" * 60)
        self.accelerator.print("PHASE 3: FINE-TUNING (+ LoRA backbone layers)")
        self.accelerator.print("=" * 60)
        
        opt_p3 = self._setup_optimizer(phase=3)
        total_steps_p3 = epochs_p3 * len(self.loader_joint)
        sched_p3 = get_scheduler(
            "cosine", optimizer=opt_p3,
            num_warmup_steps=min(100, total_steps_p3 // 10),
            num_training_steps=total_steps_p3,
        )
        
        opt_p3 = self.accelerator.prepare(opt_p3)
        
        patience = 3
        best_ft_loss = float("inf")
        
        for ep in range(epochs_p3):
            self.accelerator.print(f"\n--- Phase 3, Epoch {ep + 1}/{epochs_p3} ---")
            m = self._train_epoch(
                self.loader_joint, opt_p3, sched_p3, ep, "3-Finetune", target_pathway="Joint"
            )
            all_metrics.append({"phase": 3, "epoch": ep, "finetune": m})
            
            if m["avg_loss"] < best_ft_loss:
                best_ft_loss = m["avg_loss"]
                patience = 3
                if self.accelerator.is_main_process:
                    self.checkpoint_mgr.save_checkpoint(
                        self.accelerator.unwrap_model(self.model),
                        step=ep, is_best=True,
                    )
            else:
                patience -= 1
                if patience <= 0:
                    self.accelerator.print("Early stopping triggered in Phase 3.")
                    break
        
        # ==============================
        # Save final metrics
        # ==============================
        if self.accelerator.is_main_process:
            results_dir = self.config.get("paths", {}).get("results_dir", "./outputs/results")
            os.makedirs(results_dir, exist_ok=True)
            metrics_path = os.path.join(results_dir, "training_metrics.json")
            
            # Convert non-serializable values
            serializable = json.loads(json.dumps(all_metrics, default=str))
            with open(metrics_path, 'w') as f:
                json.dump(serializable, f, indent=2)
            
            self.accelerator.print(f"\nTraining metrics saved to {metrics_path}")
        
        self.accelerator.print("\n" + "=" * 60)
        self.accelerator.print("TRAINING PIPELINE COMPLETE")
        self.accelerator.print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="DualPath-VLM Trainer")
    parser.add_argument("--config_path", type=str, default="configs/base_config.yaml",
                        help="Path to training configuration YAML")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only load data and verify — do not train")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Override image directory (default: data/coco/train2017)")
    args = parser.parse_args()
    
    # Load config
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("DualPath-VLM Training Pipeline")
    print("=" * 60)
    print(f"Config: {args.config_path}")
    print(f"Backbone: {config['model']['backbone_id']}")
    print(f"Phases: {config['training']['epochs_phase1']}+{config['training']['epochs_phase2']}+{config['training']['epochs_phase3']} epochs")
    print(f"Batch size: {config['training']['batch_size']}")
    
    # Initialize processor
    model_id = config["model"]["backbone_id"]
    print(f"\nLoading processor for {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    # Initialize model
    print(f"Loading model {model_id}...")
    from model.dualpath_vlm import DualPathVLM
    
    model_cfg = config["model"]
    use_4bit = model_cfg.get("use_4bit", False)
    gradient_checkpointing = model_cfg.get("gradient_checkpointing", False)
    
    # Auto-detect: enable 4-bit if GPU has <24GB VRAM
    if not use_4bit and torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        if vram_gb < 24:
            print(f"\n[AUTO] Detected {vram_gb:.0f}GB VRAM — enabling 4-bit quantization + gradient checkpointing")
            use_4bit = True
            gradient_checkpointing = True
    
    model = DualPathVLM(
        model_id=model_id,
        device=model_cfg.get("device", "cuda"),
        use_4bit=use_4bit,
        gradient_checkpointing=gradient_checkpointing,
    )
    
    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Initialize trainer
    trainer = Trainer(model=model, processor=processor, config=config)
    
    # Setup data
    data_cfg = config.get("data", {})
    fab_jsonl = data_cfg.get("datasets", {}).get("fabrication_triplets", "data/real_fabrication_pairs.jsonl")
    om_jsonl = data_cfg.get("datasets", {}).get("omission_triplets", "data/real_omission_pairs.jsonl")
    image_dir = args.image_dir or "data/coco/train2017"
    
    trainer.setup_data(
        fab_jsonl=fab_jsonl,
        om_jsonl=om_jsonl,
        image_dir=image_dir,
    )
    
    if args.dry_run:
        print("\n[DRY RUN] Data loading verified. Exiting without training.")
        # Test one batch
        batch = next(iter(trainer.loader_f))
        print(f"  Sample batch keys: {list(batch.keys())}")
        print(f"  input_ids shape: {batch['inputs']['input_ids'].shape}")
        print(f"  pixel_values shape: {batch['inputs']['pixel_values'].shape}")
        print(f"  fabricated_tokens: {batch['fabricated_tokens'][:5]}...")
        return
    
    # Launch training
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    print(f"\nTotal training time: {elapsed / 3600:.1f} hours")


if __name__ == "__main__":
    main()
