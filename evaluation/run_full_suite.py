import os
import subprocess
import argparse

def run_command(cmd):
    print(f"Executing: {cmd}")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    subprocess.run(cmd, shell=True, check=True, env=env)

def main():
    parser = argparse.ArgumentParser(description="DualPath-VLM A*-Strong Evaluation Suite")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--checkpoint", type=str, default="outputs/dualpath_vlm_final.pt")
    parser.add_argument("--limit", type=int, default=500, help="Samples per dataset")
    parser.add_argument("--vg_json", type=str, default="data/vg_objects_val.json")
    parser.add_argument("--vg_img", type=str, default="data/vg/images")
    args = parser.parse_args()

    print("====================================================")
    print("   DualPath-VLM: Beginning Full Evaluation Suite")
    print("====================================================")

    # 1. Mechanistic Inference Run (for Drift and Gating plots)
    # We use COCO for this primary mechanistic log
    print("\n[Step 1/5] Running Mechanistic Inference Log (COCO)...")
    run_command(f"python evaluation/inference_engine.py --model_id {args.model_id} --checkpoint {args.checkpoint} --dataset coco --limit {args.limit} --out outputs/evaluation/inference_log.json")

    # 2. Hardened Benchmarks
    print("\n[Step 2/5] Running POPE (Adversarial)...")
    run_command(f"python evaluation/eval_pope.py --model_id {args.model_id} --limit {args.limit}")

    print("\n[Step 3/5] Running CHAIR (MSCOCO)...")
    run_command(f"python evaluation/eval_chair.py --model_id {args.model_id} --limit {args.limit}")

    print("\n[Step 4/5] Running Novel Object Recall (Visual Genome)...")
    run_command(f"python evaluation/eval_object_recall.py --vg_json {args.vg_json} --img_dir {args.vg_img} --limit {args.limit}")

    # 3. Visualization Generation
    print("\n[Step 5/5] Generating Research-Grade Visualizations...")
    run_command("python scripts/plot_tradeoff_curve.py")
    run_command("python scripts/verify_attention_drift.py --log outputs/evaluation/inference_log.json")
    run_command("python scripts/plot_gating_behavior.py")
    run_command("python scripts/plot_intervention_freq.py")

    print("\n====================================================")
    print("   Evaluation Complete! Results in outputs/results/")
    print("====================================================")

if __name__ == "__main__":
    main()
