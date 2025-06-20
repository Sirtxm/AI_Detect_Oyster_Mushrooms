import os
import shutil
import random
from pathlib import Path

# CONFIG
IMAGE_ROOT = 'OysterMushroom_image_dataset'
LABEL_ROOT = 'OysterMushrooms_boundingbox_annotations'
OUTPUT_ROOT = 'datasets/oyster_mushroom'
SPLIT_RATIO = 0.8

# สร้างโฟลเดอร์ output
for split in ['train', 'val']:
    os.makedirs(f'{OUTPUT_ROOT}/images/{split}', exist_ok=True)
    os.makedirs(f'{OUTPUT_ROOT}/labels/{split}', exist_ok=True)

# ค้นหารูปทั้งหมดแบบ recursive
image_paths = list(Path(IMAGE_ROOT).rglob("*.jpg"))
label_paths = list(Path(LABEL_ROOT).rglob("*.txt"))

# สร้างดิกชันนารี label index ตามชื่อ (ไม่รวม .txt)
label_dict = {p.stem: p for p in label_paths}

matched = []
for img_path in image_paths:
    img_name = img_path.stem
    # หา label ที่ "ขึ้นต้น" เหมือนชื่อรูป
    possible_matches = [k for k in label_dict if k.startswith(img_name)]
    if possible_matches:
        matched_label_path = label_dict[possible_matches[0]]
        matched.append((img_path, matched_label_path))
    else:
        print(f"❌ ไม่มี label สำหรับ: {img_name}")

print(f"✅ พบรูปที่จับคู่ได้ทั้งหมด: {len(matched)} / {len(image_paths)}")

# สุ่มแบ่ง train / val
random.shuffle(matched)
split_idx = int(len(matched) * SPLIT_RATIO)
train, val = matched[:split_idx], matched[split_idx:]

def move(matched_pairs, split):
    for img, lbl in matched_pairs:
        new_img_path = Path(OUTPUT_ROOT) / 'images' / split / img.name
        new_lbl_path = Path(OUTPUT_ROOT) / 'labels' / split / img.with_suffix('.txt').name
        shutil.copy(img, new_img_path)
        shutil.copy(lbl, new_lbl_path)

move(train, 'train')
move(val, 'val')

# สร้าง data.yaml
with open(f"{OUTPUT_ROOT}/data.yaml", "w") as f:
    f.write(f"""path: {OUTPUT_ROOT}
train: images/train
val: images/val

nc: 1
names: ['oyster_mushroom']
""")

print("🎉 Dataset พร้อมแล้วสำหรับ YOLOv8!")
