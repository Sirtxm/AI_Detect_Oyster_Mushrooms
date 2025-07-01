import os
import shutil
import random
from pathlib import Path

# CONFIGURATION
IMAGE_ROOT = 'OysterMushroom_image_dataset'
LABEL_ROOT = 'OysterMushrooms_boundingbox_annotations'
OUTPUT_ROOT = 'datasets/oyster_mushroom'
SPLIT_RATIO = 0.8  # 80% for training, 20% for validation

# Create output directories
for split in ['train', 'val']:
    os.makedirs(f'{OUTPUT_ROOT}/images/{split}', exist_ok=True)
    os.makedirs(f'{OUTPUT_ROOT}/labels/{split}', exist_ok=True)

# Recursively find all image and label files
image_paths = list(Path(IMAGE_ROOT).rglob("*.jpg"))
label_paths = list(Path(LABEL_ROOT).rglob("*.txt"))

# Build a dictionary of label files indexed by filename (without extension)
label_dict = {p.stem: p for p in label_paths}

# Match images with corresponding label files
matched = []
for img_path in image_paths:
    img_name = img_path.stem
    possible_matches = [k for k in label_dict if k.startswith(img_name)]
    if possible_matches:
        matched_label_path = label_dict[possible_matches[0]]
        matched.append((img_path, matched_label_path))
    else:
        print(f"❌ No label found for image: {img_name}")

print(f"✅ Matched image-label pairs: {len(matched)} / {len(image_paths)}")

# Shuffle and split into train and validation sets
random.shuffle(matched)
split_idx = int(len(matched) * SPLIT_RATIO)
train, val = matched[:split_idx], matched[split_idx:]

# Copy matched files to their corresponding directories
def move(matched_pairs, split):
    for img, lbl in matched_pairs:
        new_img_path = Path(OUTPUT_ROOT) / 'images' / split / img.name
        new_lbl_path = Path(OUTPUT_ROOT) / 'labels' / split / img.with_suffix('.txt').name
        shutil.copy(img, new_img_path)
        shutil.copy(lbl, new_lbl_path)

move(train, 'train')
move(val, 'val')

# Create data.yaml file for YOLOv8
with open(f"{OUTPUT_ROOT}/data.yaml", "w") as f:
    f.write(f"""path: {OUTPUT_ROOT}
train: images/train
val: images/val

nc: 1
names: ['oyster_mushroom']
""")

print("🎉 Dataset is ready for YOLOv8 training!")
