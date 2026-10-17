"""
download_coco_balanced.py - Tai va can bang du lieu COCO-2017 (Class-Balanced Sampling).

Muc tieu:
- Moi lop lay khoang 2.500 anh (Laptop, Cell Phone, Backpack, Person).
- Khu trung lap anh, bao dam moi anh duoc gan nhan day du cho ca 4 lop.
- Chia ngau nhien: 70% Train, 20% Val, 10% Test (Seed 42).
- Xuat du lieu chuan YOLOv5/YOLOv8 vao thu muc ./dataset_coco_custom.
"""
import json
import random
import shutil
import sys
from pathlib import Path

# Dam bao in tieng Viet tren Windows khong bi loi encode
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.random as four

# 4 lop muc tieu
TARGET_CLASSES = ["person", "laptop", "cell phone", "backpack"]

# So luong anh mong muon cho moi lop
SAMPLES_PER_CLASS = {
    "laptop": 2500,
    "cell phone": 2500,
    "backpack": 2500,
    "person": 2500,
}

SEED = 42
EXPORT_DIR = Path("./dataset_coco_custom")


def ensure_coco_annotations():
    """Dam bao file chu thich COCO instances_train2017.json da co trong may."""
    try:
        coco_path = foz.find_zoo_dataset("coco-2017")
        if coco_path:
            ann_file = Path(coco_path) / "raw" / "instances_train2017.json"
            if ann_file.exists():
                return ann_file
    except Exception:
        pass

    print(">>> Dang tai file chu thich COCO 2017 (chi chay 1 lan duy nhat)...")
    foz.download_zoo_dataset(
        "coco-2017",
        split="train",
        label_types=["detections"],
        max_samples=1,
    )
    coco_path = foz.find_zoo_dataset("coco-2017")
    return Path(coco_path) / "raw" / "instances_train2017.json"


def select_balanced_image_ids(ann_file):
    """Chon loc danh sach Image ID can bang cho 4 lop."""
    print(">>> Dang doc thong tin nhan de loc mau can bang...")
    with open(ann_file, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # Lay ID cua 4 lop
    cat_map = {
        c["id"]: c["name"]
        for c in coco_data["categories"]
        if c["name"] in TARGET_CLASSES
    }

    # Gom danh sach anh theo tung lop
    class_to_images = {name: set() for name in TARGET_CLASSES}
    for ann in coco_data["annotations"]:
        cid = ann["category_id"]
        if cid in cat_map:
            cname = cat_map[cid]
            class_to_images[cname].add(ann["image_id"])

    # Lay mau can bang ngau nhien
    rng = random.Random(SEED)
    selected_ids = set()

    print("\n--- KE HOACH LAY MAU ---")
    for cname in TARGET_CLASSES:
        target_count = SAMPLES_PER_CLASS.get(cname, 2500)
        available_imgs = sorted(list(class_to_images[cname]))
        rng.shuffle(available_imgs)
        chosen = available_imgs[:target_count]
        print(f"   {cname:<11}: co {len(available_imgs):>5} anh trong COCO -> Chon lay {len(chosen):>5} anh")
        # Format ID anh thanh chuoi 12 chu so theo chuan COCO
        for img_id in chosen:
            selected_ids.add(str(img_id).zfill(12))

    print(f"\n>>> Tong so anh duy nhat sau khi gop va khu trung lap: {len(selected_ids)} anh.")
    return sorted(list(selected_ids))


def main():
    ann_file = ensure_coco_annotations()
    selected_image_ids = select_balanced_image_ids(ann_file)

    # 1. Tai va nap dataset FiftyOne
    print("\n>>> Dang tai/nap anh va nhan vao FiftyOne...")
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="train",
        label_types=["detections"],
        classes=TARGET_CLASSES,
        image_ids=selected_image_ids,
        only_matching=True,
    )

    # 2. Xoa cac tag mac dinh truoc khi chia tap
    dataset.untag_samples(dataset.distinct("tags"))

    # 3. Phan chia tap du lieu 70% Train, 20% Val, 10% Test
    print(">>> Dang phan chia ngau nhien: 70% Train / 20% Val / 10% Test...")
    four.random_split(dataset, {"train": 0.7, "val": 0.2, "test": 0.1}, seed=SEED)

    # 4. Lam sach thu muc cu truoc khi xuat
    if EXPORT_DIR.exists():
        print(f">>> Xoa thu muc cu {EXPORT_DIR} de xuat du lieu moi hoan toan...")
        shutil.rmtree(EXPORT_DIR)

    # 5. Xuat du lieu ra format YOLOv5/YOLOv8
    print(f">>> Dang xuat du lieu ra {EXPORT_DIR} theo format YOLO...")
    for split in ["train", "val", "test"]:
        dataset.match_tags(split).export(
            export_dir=str(EXPORT_DIR),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            split=split,
            classes=TARGET_CLASSES,
        )

    # 6. Thong ke ket qua
    print("\nHoan tat xuat du lieu can bang!")
    print("Thong ke so luong nhan theo tung lop trong toan bo dataset:")
    print(dataset.count_values("ground_truth.detections.label"))


if __name__ == "__main__":
    main()
