#%%
import os
import shutil
import random

# Fix random seed for reproducibility
random.seed(42)

# Italian folder names mapped to English class names
italian_to_english = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "scoiattolo": "squirrel",
    "ragno": "spider"
}

# Source dataset path
src_root = "/Users/qianrunchen/Downloads/archive/raw-img"

# Destination path for the new subset
dst_root = "/Users/qianrunchen/Downloads/animal_subset"
os.makedirs(dst_root, exist_ok=True)

# Number of images to select per class (within 100–150 as required)
NUM_PER_CLASS = 120

total_images = 0

for ita_name, eng_name in italian_to_english.items():
    ita_folder = os.path.join(src_root, ita_name)

    if not os.path.isdir(ita_folder):
        print(f"Skipping: folder not found -> {ita_folder}")
        continue

    # List all files in the category
    all_files = [
        f for f in os.listdir(ita_folder)
        if os.path.isfile(os.path.join(ita_folder, f))
    ]

    if len(all_files) == 0:
        print(f"Skipping: no images found in {ita_folder}")
        continue

    # Shuffle file list
    random.shuffle(all_files)

    # Select up to NUM_PER_CLASS images
    n_select = min(NUM_PER_CLASS, len(all_files))
    selected_files = all_files[:n_select]

    # Create destination folder
    dst_folder = os.path.join(dst_root, eng_name)
    os.makedirs(dst_folder, exist_ok=True)

    # Copy selected images
    for fname in selected_files:
        src_path = os.path.join(ita_folder, fname)
        dst_path = os.path.join(dst_folder, fname)
        shutil.copy(src_path, dst_path)

    total_images += n_select
    print(f"Selected {n_select} images for class: {eng_name}")

print("\nSampling complete.")
print(f"New dataset location: {dst_root}")
print(f"Total number of selected images: {total_images}")


#%%
import os
import shutil
import random

SRC = "/Users/qianrunchen/Downloads/animal_subset"          # raw data
DST = "/Users/qianrunchen/Downloads/animal_subset_split"    # new

random.seed(42)

# just contain the figures
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# train/val/test
ratios = {"train": 0.7, "val": 0.15, "test": 0.15}

# target folder
for split in ratios:
    os.makedirs(os.path.join(DST, split), exist_ok=True)

# read 
categories = [
    d for d in os.listdir(SRC)
    if os.path.isdir(os.path.join(SRC, d)) and not d.startswith(".")
]

print("Detected classes:", categories)

for cls in categories:
    src_dir = os.path.join(SRC, cls)

    # images
    images = [
        f for f in os.listdir(src_dir)
        if f.lower().endswith(VALID_EXT)
    ]

    if len(images) == 0:
        print(f"[WARNING] class '{cls}' has NO images, skipping!")
        continue

    total = len(images)
    random.shuffle(images)

    n_train = int(total * ratios["train"])
    n_val = int(total * ratios["val"])
    n_test = total - n_train - n_val

    split_files = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split_name, file_list in split_files.items():
        target_dir = os.path.join(DST, split_name, cls)
        os.makedirs(target_dir, exist_ok=True)

        for fname in file_list:
            shutil.copy2(os.path.join(src_dir, fname),
                         os.path.join(target_dir, fname))

print("\nDone!")
print("Your dataset is ready at:", DST)

#%%
# ========================================================
# DEBUG: Check the first image in training set
# ========================================================
def debug_quick_check():
    train_dir = os.path.join(DATA_ROOT, "train")

    # find first valid image
    sample_path = None
    for cls in os.listdir(train_dir):
        if cls.startswith("."):
            continue
        cls_folder = os.path.join(train_dir, cls)
        if not os.path.isdir(cls_folder):
            continue
        
        for fname in os.listdir(cls_folder):
            if fname.lower().endswith(("jpg","jpeg","png")):
                sample_path = os.path.join(cls_folder, fname)
                break
        if sample_path:
            break

    if not sample_path:
        print("No image found in training set.")
        return

    print("Sample image path:", sample_path)

    # Load original image
    img = Image.open(sample_path).convert("RGB")
    w, h = img.size
    print("Original size:", (w, h))

    # Apply your preprocessing
    ds = CustomImageDataset(os.path.join(DATA_ROOT, "train"))
    tensor = ds.load_image(sample_path)

    print("Tensor shape:", tensor.shape)
    print("Tensor min/max:", tensor.min().item(), tensor.max().item())
    print("Tensor mean:", tensor.mean().item())

    # Print the first 20 pixel numbers so you can see if it's noise
    flat_first = tensor.flatten()[:20]
    print("First 20 tensor values:", flat_first.tolist())


if __name__ == "__main__":
    debug_quick_check()


#%%
def debug_count_classes():
    root = os.path.join(DATA_ROOT, "train")
    print("Train class distribution:")

    for cls in sorted(os.listdir(root)):
        if cls.startswith("."):
            continue
        cls_dir = os.path.join(root, cls)
        if os.path.isdir(cls_dir):
            imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(("jpg","jpeg","png"))]
            print(cls, ":", len(imgs))

debug_count_classes()
