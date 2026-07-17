"""
DualPath-VLM: Automated Dataset Downloader for Cloud/Local Training.
Downloads COCO 2017 (annotations + images) and Visual Genome objects.

Usage:
    python scripts/download_cloud_data.py                    # Full download (~20GB)
    python scripts/download_cloud_data.py --val-only         # Val images only (~1GB)
    python scripts/download_cloud_data.py --annotations-only # Annotations only (~300MB)
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import hashlib
from tqdm import tqdm

# --- Download Manifest ---
DOWNLOADS = {
    "coco_annotations": {
        "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "extract_to": "data/coco",
        "description": "COCO 2017 Annotations (instances + captions)",
        "size_mb": 241,
    },
    "coco_train_images": {
        "url": "http://images.cocodataset.org/zips/train2017.zip",
        "extract_to": "data/coco",
        "description": "COCO 2017 Training Images (118K images)",
        "size_mb": 18000,
    },
    "coco_val_images": {
        "url": "http://images.cocodataset.org/zips/val2017.zip",
        "extract_to": "data/coco",
        "description": "COCO 2017 Validation Images (5K images)",
        "size_mb": 780,
    },
    "coco_captions": {
        "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "extract_to": "data/coco",
        "description": "COCO 2017 Captions (included in annotations zip)",
        "size_mb": 0,  # Already included in coco_annotations
        "skip": True,  # Same zip as coco_annotations
    },
    "visual_genome_objects": {
        "url": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip",
        "extract_to": "data/vg",
        "description": "Visual Genome Objects v1.4",
        "size_mb": 30,
    },
}


class DownloadProgressBar(tqdm):
    """tqdm wrapper for urllib downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest_path: str, description: str):
    """Downloads a file with a progress bar."""
    if os.path.exists(dest_path):
        print(f"  [SKIP] {dest_path} already exists.")
        return
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"  Downloading: {description}")
    print(f"  URL: {url}")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(dest_path)) as t:
        urllib.request.urlretrieve(url, dest_path, reporthook=t.update_to)
    
    print(f"  [OK] Saved to {dest_path}")


def extract_zip(zip_path: str, extract_to: str):
    """Extracts a zip file."""
    print(f"  Extracting {zip_path} -> {extract_to}/ ...")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)
    print(f"  [OK] Extraction complete.")


def verify_expected_files(base_dir: str):
    """Checks that all required files exist after download."""
    expected = {
        "COCO instances (train)": "data/coco/annotations/instances_train2017.json",
        "COCO instances (val)": "data/coco/annotations/instances_val2017.json",
        "COCO captions (train)": "data/coco/annotations/captions_train2017.json",
        "COCO captions (val)": "data/coco/annotations/captions_val2017.json",
    }
    
    print("\n--- File Integrity Check ---")
    all_ok = True
    for name, path in expected.items():
        full_path = os.path.join(base_dir, path)
        if os.path.exists(full_path):
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            print(f"  [✓] {name}: {full_path} ({size_mb:.1f} MB)")
        else:
            print(f"  [✗] {name}: MISSING at {full_path}")
            all_ok = False
    
    # Check for images directory
    for img_dir in ["data/coco/train2017", "data/coco/val2017"]:
        full_path = os.path.join(base_dir, img_dir)
        if os.path.isdir(full_path):
            count = len([f for f in os.listdir(full_path) if f.endswith('.jpg')])
            print(f"  [✓] {img_dir}: {count} images")
        else:
            print(f"  [?] {img_dir}: not found (may not have been requested)")
    
    # Check VG
    vg_path = os.path.join(base_dir, "data/vg/objects.json")
    if os.path.exists(vg_path):
        size_mb = os.path.getsize(vg_path) / (1024 * 1024)
        print(f"  [✓] Visual Genome objects: {vg_path} ({size_mb:.1f} MB)")
    else:
        print(f"  [?] Visual Genome objects: not found")
    
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="DualPath-VLM Data Downloader")
    parser.add_argument("--val-only", action="store_true", 
                        help="Download only validation images (~1GB) instead of full training set (~18GB)")
    parser.add_argument("--annotations-only", action="store_true",
                        help="Download only annotations (~300MB), no images")
    parser.add_argument("--base-dir", type=str, default=".",
                        help="Base directory for the project (default: current directory)")
    args = parser.parse_args()
    
    base_dir = os.path.abspath(args.base_dir)
    
    print("=" * 60)
    print("DualPath-VLM: Automated Dataset Downloader")
    print("=" * 60)
    
    # Determine what to download
    download_keys = ["coco_annotations", "visual_genome_objects"]
    
    if not args.annotations_only:
        if args.val_only:
            download_keys.append("coco_val_images")
            print("\nMode: Validation images only (~1GB total)")
        else:
            download_keys.extend(["coco_train_images", "coco_val_images"])
            print("\nMode: Full dataset (~20GB total)")
    else:
        print("\nMode: Annotations only (~300MB total)")
    
    # Execute downloads
    for key in download_keys:
        info = DOWNLOADS[key]
        if info.get("skip"):
            continue
        
        print(f"\n--- {info['description']} ---")
        
        zip_filename = info["url"].split("/")[-1]
        zip_path = os.path.join(base_dir, "data", "downloads", zip_filename)
        extract_to = os.path.join(base_dir, info["extract_to"])
        
        download_file(info["url"], zip_path, info["description"])
        extract_zip(zip_path, extract_to)
    
    # Verify
    verify_expected_files(base_dir)
    
    print("\n" + "=" * 60)
    print("Download complete! Next step:")
    print("  python scripts/generate_real_pairs.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
