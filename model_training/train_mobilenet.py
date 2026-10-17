"""
train_mobilenet.py - Huan luyen SSDLite320-MobileNetV3 tren GPU RTX 3050.
Kien truc dung Depthwise Separable Convolution theo dung thuyet minh Seminar.
"""
import os
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from PIL import Image

# 4 lop doi tuong (Khop voi dataset.yaml)
# Luu y: PyTorch Detection danh index 0 cho Background, 4 lop danh so tu 1 den 4
CLASSES = ["background", "person", "laptop", "cell phone", "backpack"]
NUM_CLASSES = len(CLASSES)  # 5 classes (1 background + 4 objects)


import random


class YOLODataset(Dataset):
    """Doc truc tiep du lieu anh va nhan YOLO tu thu muc dataset_coco_custom, kem Data Augmentation."""
    def __init__(self, root_dir, split="train", img_size=320):
        self.img_dir = Path(root_dir) / "images" / split
        self.lbl_dir = Path(root_dir) / "labels" / split
        self.split = split
        self.img_size = img_size
        self.images = sorted(list(self.img_dir.glob("*.jpg")))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        lbl_path = self.lbl_dir / (img_path.stem + ".txt")

        # 1. Doc va resize anh ve 320x320
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)

        # DATA AUGMENTATION (Chi ap dung khi train, khong ap dung cho val/test)
        do_flip = (self.split == "train") and (random.random() < 0.5)
        if do_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Bien doi do sang & tuong phan nhe (gia lap anh sang phong hoc)
        if self.split == "train" and random.random() < 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
            img = torchvision.transforms.functional.adjust_brightness(img, brightness_factor)
            img = torchvision.transforms.functional.adjust_contrast(img, contrast_factor)

        # Chuyen anh sang Tensor va chuan hoa [0.0, 1.0]
        img_tensor = torchvision.transforms.functional.to_tensor(img)

        # 2. Doc toa do Bounding Box tu file nhan txt cua YOLO
        boxes = []
        labels = []
        if lbl_path.exists():
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])

                    # Neu anh bi lat ngang thi dao nguoc tam cx
                    if do_flip:
                        cx = 1.0 - cx

                    # Chuyen doi tu YOLO (cx, cy, w, h) sang Pixel Coordinates (x1, y1, x2, y2)
                    x1 = (cx - w / 2) * self.img_size
                    y1 = (cy - h / 2) * self.img_size
                    x2 = (cx + w / 2) * self.img_size
                    y2 = (cy + h / 2) * self.img_size

                    # Kiem tra hop le
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(cls_id + 1)  # +1 vi class 0 la Background

        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }
        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f">>> Su dung thiet bi: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Khoi tao Dataset & DataLoader
    dataset_root = "dataset_coco_custom"
    train_dataset = YOLODataset(dataset_root, split="train", img_size=320)
    val_dataset = YOLODataset(dataset_root, split="val", img_size=320)

    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True,
        num_workers=2, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False,
        num_workers=2, collate_fn=collate_fn
    )

    # 2. Khoi tao Model SSDLite320-MobileNetV3
    print(">>> Dang tai kien truc SSDLite320-MobileNetV3 (Depthwise Separable Conv)...")
    model = ssdlite320_mobilenet_v3_large(
        num_classes=NUM_CLASSES,
        weights_backbone="DEFAULT"  # Dung trong so MobileNetV3 goc tu ImageNet
    )
    model.to(device)

    # 3. Thuat toan toi uu (Optimizer & Scheduler)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.0005, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # 4. Thu muc luu ket qua
    save_dir = Path("runs_mobilenet")
    save_dir.mkdir(exist_ok=True)
    best_loss = float("inf")
    EPOCHS = 30  # 30 epoch la rat dep cho MobileNet

    print("\n" + "="*60)
    print(">>> BAT DAU HUAN LUYEN SSDLite320-MobileNetV3")
    print("="*60)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"Epoch [{epoch}/{EPOCHS}] - Batch [{batch_idx+1}/{len(train_loader)}] - Loss: {losses.item():.4f}")

        lr_scheduler.step()
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_time

        print(f"\n--- KET THUC EPOCH {epoch} ({epoch_time:.1f}s) | Avg Loss: {avg_loss:.4f} ---")

        # Luu model tot nhat
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = save_dir / "best_mobilenet.pth"
            torch.save(model.state_dict(), save_path)
            print(f">>> Da luu checkpoint tot nhat vao: {save_path}\n")

    print("\n✅ HOAN TAT HUAN LUYEN MOBILENET!")
    print(f"Model duoc luu tai: {save_dir / 'best_mobilenet.pth'}")


if __name__ == "__main__":
    train()
