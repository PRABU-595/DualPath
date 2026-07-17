"""
DualPath-Unified: Comprehensive Text Hallucination Dataset Downloader
Downloads and processes TruthfulQA, FaithDial, and CNN/DailyMail summarization
into DualPath JSONL format, balancing the LLM text split against COCO (118K images).

Output:
    D:\\Dataset\\DualPath_Text_Data\\text_fabrication_pairs.jsonl  (combined)
    D:\\Dataset\\DualPath_Text_Data\\text_omission_pairs.jsonl     (combined)
"""

import json
import os
import sys
import subprocess

OUTPUT_DIR = r"D:\Dataset\DualPath_Text_Data"

# ─────────────────────────────────────────────────────────────────────────────
def ensure_deps():
    try:
        import datasets
    except ImportError:
        print("[setup] Installing 'datasets' library...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "-q"])

# ─────────────────────────────────────────────────────────────────────────────
def process_halueval(fab_out, om_out):
    """Already downloaded — reads existing HaluEval pairs (5K+5K)."""
    from datasets import load_dataset
    print("\n[1/3] HaluEval (QA) ...")
    fab, om = [], []
    try:
        ds = load_dataset("pminervini/HaluEval", "qa", split="data")
        for i, item in enumerate(ds):
            ctx  = item.get("knowledge", "")
            q    = item.get("question", "")
            good = item.get("right_answer", "")
            bad  = item.get("hallucinated_answer", "")
            prompt = f"Context: {ctx}\nQuestion: {q}"
            if i % 2 == 0:
                fab.append({"prompt": prompt, "good_caption": good,
                            "bad_caption": bad,
                            "fabricated_objects": ["halueval_fabrication"],
                            "pathway": "F"})
            else:
                words = good.split()
                om_ans = " ".join(words[:max(1,len(words)//2)]) + "..." if len(words)>4 else "I don't know."
                om.append({"prompt": prompt, "good_caption": good,
                           "bad_caption": om_ans,
                           "omitted_objects": ["halueval_omission"],
                           "pathway": "O"})
        print(f"  → {len(fab)} fabrication, {len(om)} omission")
    except Exception as e:
        print(f"  [WARN] HaluEval failed: {e}")
    return fab, om

# ─────────────────────────────────────────────────────────────────────────────
def process_truthfulqa():
    """TruthfulQA: question → best_answer (good) vs incorrect_answers[0] (bad)."""
    from datasets import load_dataset
    print("\n[2/4] TruthfulQA ...")
    fab, om = [], []
    try:
        ds = load_dataset("truthful_qa", "generation", split="validation")
        for item in ds:
            q    = item.get("question", "")
            good = item.get("best_answer", "")
            bads = item.get("incorrect_answers", [])
            if not good or not bads:
                continue
            bad = bads[0]
            prompt = f"Question: {q}"
            # TruthfulQA is all about fabrication (false confident answers)
            fab.append({"prompt": prompt, "good_caption": good,
                        "bad_caption": bad,
                        "fabricated_objects": ["truthfulqa_hallucination"],
                        "pathway": "F"})
            # Simulate omission: only say "I'm not certain" → missing key facts
            words = good.split()
            om_ans = " ".join(words[:max(1,len(words)//2)]) + "..." if len(words)>4 else "I'm not certain."
            om.append({"prompt": prompt, "good_caption": good,
                       "bad_caption": om_ans,
                       "omitted_objects": ["truthfulqa_omission"],
                       "pathway": "O"})
        print(f"  → {len(fab)} fabrication, {len(om)} omission")
    except Exception as e:
        print(f"  [WARN] TruthfulQA failed: {e}")
    return fab, om

# ─────────────────────────────────────────────────────────────────────────────
def process_faithdial():
    """Use 'MuskumPillerum/General-Knowledge' Q&A or fall back to
    'Helsinki-NLP/open_subtitles' — replaced with 'wiki_qa' which has
    question + document grounding to simulate faithfulness."""
    from datasets import load_dataset
    print("\n[3/4] WikiQA (faithfulness grounding) ...")
    fab, om = [], []    
    try:
        ds = load_dataset("wiki_qa", split="train", trust_remote_code=True)
        for item in ds:
            q       = item.get("question", "")
            doc     = item.get("document_title", "")
            answer  = item.get("answer", "")
            label   = item.get("label", 0)
            if not q or not answer:
                continue
            prompt = f"Document: {doc}\nQuestion: {q}"
            if label == 1:   # Correct answer = good; fabricate a wrong answer
                bad = answer + " Additionally, experts have confirmed this is universally disputed."
                fab.append({"prompt": prompt, "good_caption": answer,
                            "bad_caption": bad,
                            "fabricated_objects": ["wikiq_fabrication"],
                            "pathway": "F"})
                words = answer.split()
                om_ans = " ".join(words[:max(1,len(words)//2)]) + "..." if len(words)>3 else "Unknown."
                om.append({"prompt": prompt, "good_caption": answer,
                           "bad_caption": om_ans,
                           "omitted_objects": ["wikiq_omission"],
                           "pathway": "O"})
        print(f"  → {len(fab)} fabrication, {len(om)} omission")
    except Exception as e:
        print(f"  [WARN] WikiQA failed: {e}")
    return fab, om

# ─────────────────────────────────────────────────────────────────────────────
def process_cnn_dailymail(max_samples=60000):
    """XSum: extreme summarization — article → one-sentence summary.
    Uses streaming so only the needed samples are fetched."""
    from datasets import load_dataset
    print(f"\n[4/4] XSum summarization (up to {max_samples} samples, streaming) ...")
    fab, om = [], []
    try:
        ds = load_dataset("EdinburghNLP/xsum", split="train", streaming=True,
                          trust_remote_code=False)
        count = 0
        for item in ds:
            if count >= max_samples:
                break
            doc     = item.get("document", "")
            summary = item.get("summary", "")
            if not doc or not summary:
                continue
            doc_short = " ".join(doc.split()[:200])
            prompt = f"Article: {doc_short}\nWrite a one-sentence summary."

            # Fabrication: add unsupported claim to the correct summary
            bad_fab = summary + " However, no official confirmation has been given."
            fab.append({"prompt": prompt, "good_caption": summary,
                        "bad_caption": bad_fab,
                        "fabricated_objects": ["xsum_fabrication"],
                        "pathway": "F"})
            # Omission: truncate summary to simulate ignoring half the article
            words = summary.split()
            om_ans = " ".join(words[:max(1,len(words)//2)]) + "..." if len(words)>4 else summary[:40]+"..."
            om.append({"prompt": prompt, "good_caption": summary,
                       "bad_caption": om_ans,
                       "omitted_objects": ["xsum_omission"],
                       "pathway": "O"})
            count += 1
            if count % 10000 == 0:
                print(f"  ... {count}/{max_samples} samples")
        print(f"  → {len(fab)} fabrication, {len(om)} omission")
    except Exception as e:
        print(f"  [WARN] XSum failed: {e}")
    return fab, om

# ─────────────────────────────────────────────────────────────────────────────
def save(fab_list, om_list):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fab_path = os.path.join(OUTPUT_DIR, "text_fabrication_pairs.jsonl")
    om_path  = os.path.join(OUTPUT_DIR, "text_omission_pairs.jsonl")

    with open(fab_path, "w", encoding="utf-8") as f:
        for p in fab_list:
            f.write(json.dumps(p) + "\n")

    with open(om_path, "w", encoding="utf-8") as f:
        for p in om_list:
            f.write(json.dumps(p) + "\n")

    print(f"\n{'='*60}")
    print(f"  Saved {len(fab_list):,} fabrication pairs → {fab_path}")
    print(f"  Saved {len(om_list):,} omission pairs    → {om_path}")
    print(f"  Grand total: {len(fab_list)+len(om_list):,} text training pairs")
    print(f"  Target (to match COCO 118K): ~118,287 per pathway")
    print(f"{'='*60}\n")

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ensure_deps()

    all_fab, all_om = [], []

    # 1. HaluEval QA (10K)
    f, o = process_halueval(None, None)
    all_fab.extend(f); all_om.extend(o)

    # 2. TruthfulQA (~800 → both F and O)
    f, o = process_truthfulqa()
    all_fab.extend(f); all_om.extend(o)

    # 3. FaithDial (~50K dialogue pairs)
    f, o = process_faithdial()
    all_fab.extend(f); all_om.extend(o)

    # 4. CNN/DailyMail (fill up to ~60K to reach ~118K total)
    remaining = max(0, 118287 - len(all_fab))
    f, o = process_cnn_dailymail(max_samples=remaining)
    all_fab.extend(f); all_om.extend(o)

    save(all_fab, all_om)
