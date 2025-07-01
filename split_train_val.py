import os
import shutil
import random
from pathlib import Path

base_path = "datasets/oyster_mushroom"
train_img = Path(base_path) / "images/train"
train_lbl = Path(base_path) / "labels/train"
val_img = Path(base_path) / "images/val"
val_lbl = Path(base_path) / "labels/val"

val_img.mkdir(parents=True, exist_ok=True)
val_lbl.mkdir(parents=True, exist_ok=True)

images = [f for f in train_img.iterdir() if f.suffix == '.jpg']
random.shuffle(images)
val_count = int(len(images) * 0.2)
val_images = images[:val_count]

for img_path in val_images:
    lbl_path = train_lbl / img_path.with_suffix(".txt").name
    shutil.move(str(img_path), val_img / img_path.name)
    if lbl_path.exists():
        shutil.move(str(lbl_path), val_lbl / lbl_path.name)

print(f"✅ : {val_count} ")
