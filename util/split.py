import os
import random
import shutil

random.seed(0)
# --- Configuration ---
source_dir = '/data/imagenet/train'
# Set k to the number of subdirs you want to sample
k = 50

# --- Script ---
# 1. Find all immediate subdirectories
try:
    all_entries = os.listdir(source_dir)
    all_subdirs = [
        os.path.join(source_dir, d) 
        for d in all_entries 
        if os.path.isdir(os.path.join(source_dir, d))
    ]
except FileNotFoundError:
    print(f"Error: Directory not found at {source_dir}")
    exit(1)
except NotADirectoryError:
    print(f"Error: Path is not a directory: {source_dir}")
    exit(1)

if not all_subdirs:
    print(f"No subdirectories found in {source_dir}")
    exit(0)

# 2. Check if k is valid
if k > len(all_subdirs):
    print(f"Warning: You asked for {k} samples, but only {len(all_subdirs)} subdirectories exist.")
    k = len(all_subdirs) # Sample all of them

# 3. Sample k random subdirectories
sampled_dirs = random.sample(all_subdirs, k)

# 4. Just print the list
print(f"--- Sampled {k} directories: ---")
for d in sampled_dirs:
    print(d)

# --- Optional: Uncomment to create symlinks ---
target_dir = '/data/imagenet_50_private/train'
os.makedirs(target_dir, exist_ok=True)
print(f"\n--- Creating symlinks in {target_dir} ---")
for source_path in sampled_dirs:
    dir_name = os.path.basename(source_path)
    link_path = os.path.join(target_dir, dir_name)
    try:
        if not os.path.exists(link_path):
            os.symlink(source_path, link_path)
            print(f"Created link: {link_path} -> {source_path}")
        else:
            print(f"Link already exists: {link_path}")
    except OSError as e:
        print(f"Error creating link {link_path}: {e}")


target_dir = '/data/imagenet_50_private/val'
os.makedirs(target_dir, exist_ok=True)
print(f"\n--- Creating symlinks in {target_dir} ---")
for source_path in sampled_dirs:
    source_path = source_path.replace("train", "val")
    dir_name = os.path.basename(source_path)
    link_path = os.path.join(target_dir, dir_name)
    try:
        if not os.path.exists(link_path):
            os.symlink(source_path, link_path)
            print(f"Created link: {link_path} -> {source_path}")
        else:
            print(f"Link already exists: {link_path}")
    except OSError as e:
        print(f"Error creating link {link_path}: {e}")