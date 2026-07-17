import os
import shutil
import zipfile

source_dir = "zip"
target_dir = "DualPath_VLM_Organized"

print("1/3 Creating clean directory structure...")
os.makedirs(target_dir, exist_ok=True)
os.makedirs(f"{target_dir}/Code", exist_ok=True)
os.makedirs(f"{target_dir}/Figures", exist_ok=True)
os.makedirs(f"{target_dir}/Evaluation_Logs", exist_ok=True)
os.makedirs(f"{target_dir}/Model_Checkpoints", exist_ok=True)

print("2/3 Arranging files logically...")
# 1. Move Code
if os.path.exists(f"{source_dir}/scripts"):
    for file in os.listdir(f"{source_dir}/scripts"):
        if file.endswith('.py') or file.endswith('.sh'):
            shutil.copy(f"{source_dir}/scripts/{file}", f"{target_dir}/Code/{file}")

# 2. Move Figures (Flattens nested images)
if os.path.exists(f"{source_dir}/outputs/results"):
    for root, dirs, files in os.walk(f"{source_dir}/outputs/results"):
        for file in files:
            if file.endswith('.png') or file.endswith('.pdf'):
                shutil.copy(os.path.join(root, file), f"{target_dir}/Figures/{file}")

# 3. Move Logs
if os.path.exists(f"{source_dir}/outputs/evaluation"):
    for file in os.listdir(f"{source_dir}/outputs/evaluation"):
        if file.endswith('.json'):
            shutil.copy(f"{source_dir}/outputs/evaluation/{file}", f"{target_dir}/Evaluation_Logs/{file}")

# 4. Move Model Checkpoints
if os.path.exists(f"{source_dir}/outputs/checkpoints/best_model"):
    for root, dirs, files in os.walk(f"{source_dir}/outputs/checkpoints/best_model"):
        for file in files:
            # maintain subdirectories for checkpoints if needed, but here we just copy
            rel_dir = os.path.relpath(root, f"{source_dir}/outputs/checkpoints/best_model")
            dest_dir = os.path.join(f"{target_dir}/Model_Checkpoints", rel_dir)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(os.path.join(root, file), os.path.join(dest_dir, file))

print("3/3 Packaging into DualPath_VLM_Final_Submission.zip...")
with zipfile.ZipFile("DualPath_VLM_Final_Submission.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            file_path = os.path.join(root, file)
            zf.write(file_path, arcname=os.path.relpath(file_path, target_dir))

print("✅ Organization and Zipping Complete!")
