"""
dataset_report.py - Thong ke dataset YOLO va ve anh mau co bounding box.

Cach dung (dat file nay canh thu muc dataset_coco_custom):
    pip install matplotlib pillow
    python dataset_report.py
    python dataset_report.py duong/dan/toi/dataset_coco_custom   # neu o ccho khac

Ket qua nam trong thu muc  dataset_report/ :
    - stats.csv                  : so anh, so nhan theo lop cho tung tap
    - split_images.png           : bieu do so anh 3 tap (train/val/test)
    - class_distribution.png     : bieu do so nhan theo lop va theo tap
    - samples_train.png (val/test): luoi 6 anh mau co bounding box
    - augmentation_demo.png      : minh hoa 6 buoc Data Augmentation (Letterbox 320x320, Flip, HSV, Mosaic)
    - Bao cao kiem tra loi in ra man hinh (nhan thieu, toa do sai, ...)
"""
import csv
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # khong can man hinh, chi luu file
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageEnhance, ImageOps

NAMES = ["person", "laptop", "cell phone", "backpack"]  # khop voi dataset.yaml
SPLITS = ["train", "val", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231"]
SEED = 42


def find_root():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    return Path(__file__).resolve().parent / "dataset_coco_custom"


def read_labels(label_path):
    """Tra ve (danh sach box hop le, so dong loi). Box = (cls, cx, cy, w, h)."""
    boxes, bad = [], 0
    for ln in label_path.read_text(encoding="utf-8").splitlines():
        parts = ln.split()
        if not parts:
            continue
        try:
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
        except (ValueError, IndexError):
            bad += 1
            continue
        ok = (
            0 <= cls < len(NAMES)
            and 0 <= cx <= 1 and 0 <= cy <= 1
            and 0 < w <= 1 and 0 < h <= 1
        )
        if ok:
            boxes.append((cls, cx, cy, w, h))
        else:
            bad += 1
    return boxes, bad


def analyze(root):
    results = {}
    for split in SPLITS:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        if not img_dir.exists():
            print(f"[!] Khong thay thu muc {img_dir}")
            continue

        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        class_count = Counter()
        img_with_class = Counter()  # so anh co chua it nhat 1 box cua lop
        n_boxes = n_empty = n_missing = n_bad = 0
        small_boxes = 0  # box chiem < 1% dien tich anh
        per_image = {}

        for img in images:
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                n_missing += 1
                per_image[img] = []
                continue
            boxes, bad = read_labels(lbl)
            n_bad += bad
            if not boxes:
                n_empty += 1
            per_image[img] = boxes
            for cls, _, _, w, h in boxes:
                class_count[cls] += 1
                n_boxes += 1
                if w * h < 0.01:
                    small_boxes += 1
            for cls in {b[0] for b in boxes}:
                img_with_class[cls] += 1

        # nhan khong co anh tuong ung
        img_stems = {p.stem for p in images}
        orphan = [p for p in lbl_dir.glob("*.txt") if p.stem not in img_stems] if lbl_dir.exists() else []

        results[split] = dict(
            n_images=len(images), n_boxes=n_boxes, class_count=class_count,
            img_with_class=img_with_class, n_empty=n_empty, n_missing=n_missing,
            n_bad=n_bad, small_boxes=small_boxes, orphan=len(orphan),
            per_image=per_image,
        )
    return results


def print_report(results):
    total_imgs = sum(r["n_images"] for r in results.values())
    print("=" * 64)
    print("BAO CAO DATASET")
    print("=" * 64)
    for split, r in results.items():
        pct = 100 * r["n_images"] / total_imgs if total_imgs else 0
        print(f"\n[{split.upper()}] {r['n_images']} anh ({pct:.1f}%), {r['n_boxes']} box")
        for i, name in enumerate(NAMES):
            print(f"   {name:<11}: {r['class_count'][i]:>6} box | {r['img_with_class'][i]:>5} anh")
        if r["n_boxes"]:
            print(f"   box nho (<1% dien tich anh): {r['small_boxes']} "
                  f"({100 * r['small_boxes'] / r['n_boxes']:.1f}%)")
    print("\n--- KIEM TRA LOI ---")
    issues = False
    for split, r in results.items():
        msgs = []
        if r["n_missing"]:
            msgs.append(f"{r['n_missing']} anh KHONG co file nhan")
        if r["n_empty"]:
            msgs.append(f"{r['n_empty']} file nhan rong")
        if r["n_bad"]:
            msgs.append(f"{r['n_bad']} dong nhan sai dinh dang/toa do")
        if r["orphan"]:
            msgs.append(f"{r['orphan']} file nhan khong co anh")
        if msgs:
            issues = True
            print(f"[{split}] " + "; ".join(msgs))
    if not issues:
        print("Khong phat hien loi.")


def save_csv(results, out):
    with open(out / "stats.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["split", "n_images", "n_boxes"] + [f"boxes_{n}" for n in NAMES]
                   + [f"images_{n}" for n in NAMES])
        for split, r in results.items():
            w.writerow([split, r["n_images"], r["n_boxes"]]
                       + [r["class_count"][i] for i in range(len(NAMES))]
                       + [r["img_with_class"][i] for i in range(len(NAMES))])


def plot_split(results, out):
    splits = list(results)
    counts = [results[s]["n_images"] for s in splits]
    total = sum(counts)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(splits, counts, color=["#4363d8", "#3cb44b", "#f58231"][: len(splits)])
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c}\n({100 * c / total:.1f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set_title("So anh theo tung tap")
    ax.set_ylabel("So anh")
    ax.set_ylim(0, max(counts) * 1.2)
    fig.tight_layout()
    fig.savefig(out / "split_images.png", dpi=160)
    plt.close(fig)


def plot_classes(results, out):
    splits = list(results)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / len(splits)
    for k, split in enumerate(splits):
        vals = [results[split]["class_count"][i] for i in range(len(NAMES))]
        xs = [i + k * width for i in range(len(NAMES))]
        bars = ax.bar(xs, vals, width, label=split)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks([i + width * (len(splits) - 1) / 2 for i in range(len(NAMES))])
    ax.set_xticklabels(NAMES)
    ax.set_title("So bounding box theo lop va theo tap")
    ax.set_ylabel("So box")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "class_distribution.png", dpi=160)
    plt.close(fig)


def plot_samples(results, split, out, n=6):
    per_image = results[split]["per_image"]
    candidates = [(p, b) for p, b in per_image.items() if b]
    if not candidates:
        return
    rng = random.Random(SEED)
    rng.shuffle(candidates)
    # uu tien anh co nhieu lop khac nhau de minh hoa dep hon
    candidates.sort(key=lambda x: -len({b[0] for b in x[1]}))
    top = candidates[: max(n * 3, n)]
    chosen = rng.sample(top, min(n, len(top)))

    cols = 3
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax in axes:
        ax.axis("off")
    for ax, (img_path, boxes) in zip(axes, chosen):
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        ax.imshow(img)
        for cls, cx, cy, w, h in boxes:
            x, y = (cx - w / 2) * W, (cy - h / 2) * H
            rect = patches.Rectangle((x, y), w * W, h * H, linewidth=2,
                                     edgecolor=COLORS[cls], facecolor="none")
            ax.add_patch(rect)
            ax.text(x, max(y - 3, 0), NAMES[cls], color="white", fontsize=8,
                    bbox=dict(facecolor=COLORS[cls], alpha=0.85, pad=1, edgecolor="none"))
        ax.set_title(img_path.name, fontsize=8)
    fig.suptitle(f"Anh mau tap {split} (bounding box theo nhan)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / f"samples_{split}.png", dpi=140)
    plt.close(fig)


def plot_augmentation_demo(results, out):
    """Ve bieu do minh hoa 6 ky thuat Data Augmentation / Preprocessing 320x320 cho Slide."""
    train_per_image = results.get("train", {}).get("per_image", {})
    candidates = [(p, b) for p, b in train_per_image.items() if b]
    if not candidates:
        return

    # Uu tien anh co chua nhieu doi tuong nhat de minh hoa dep
    candidates.sort(key=lambda x: -len({b[0] for b in x[1]}))
    main_img_path, main_boxes = candidates[0]

    img = Image.open(main_img_path).convert("RGB")
    W, H = img.size

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for ax in axes:
        ax.axis("off")

    def draw_yolo_boxes(ax, bxs, im_w, im_h, offset_x=0, offset_y=0, scale_w=1.0, scale_h=1.0):
        for cls, cx, cy, w, h in bxs:
            bw = w * im_w * scale_w
            bh = h * im_h * scale_h
            bx = (cx - w / 2) * im_w * scale_w + offset_x
            by = (cy - h / 2) * im_h * scale_h + offset_y
            rect = patches.Rectangle((bx, by), bw, bh, linewidth=2,
                                     edgecolor=COLORS[cls], facecolor="none")
            ax.add_patch(rect)
            ax.text(bx, max(by - 3, 0), NAMES[cls], color="white", fontsize=8,
                    bbox=dict(facecolor=COLORS[cls], alpha=0.85, pad=1, edgecolor="none"))

    # 1. Anh goc
    axes[0].imshow(img)
    draw_yolo_boxes(axes[0], main_boxes, W, H)
    axes[0].set_title(f"1. Anh goc (Original Input)\nDo phan giai goc: {W}x{H}px", fontsize=11, fontweight="bold")

    # 2. Resize & Letterbox (320x320)
    canvas_size = 320
    scale = min(canvas_size / W, canvas_size / H)
    nw, nh = int(W * scale), int(H * scale)
    resized_img = img.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (canvas_size, canvas_size), (114, 114, 114))
    dx, dy = (canvas_size - nw) // 2, (canvas_size - nh) // 2
    canvas.paste(resized_img, (dx, dy))
    axes[1].imshow(canvas)
    draw_yolo_boxes(axes[1], main_boxes, W, H, offset_x=dx, offset_y=dy, scale_w=scale, scale_h=scale)
    axes[1].set_title(f"2. Letterbox & Normalize (320x320)\nChuan hoa dau vao cho Mobile NPU", fontsize=11, fontweight="bold")

    # 3. Lat ngang (Random Horizontal Flip)
    img_flipped = ImageOps.mirror(img)
    axes[2].imshow(img_flipped)
    flipped_boxes = [(cls, 1.0 - cx, cy, w, h) for cls, cx, cy, w, h in main_boxes]
    draw_yolo_boxes(axes[2], flipped_boxes, W, H)
    axes[2].set_title("3. Lat ngang (Horizontal Flip)\nChong hoc vet huong cua vat the", fontsize=11, fontweight="bold")

    # 4. Giam sang / Doi tuong phan (Phong hoc toi / nguoc sang)
    img_dark = ImageEnhance.Brightness(img).enhance(0.55)
    img_dark = ImageEnhance.Contrast(img_dark).enhance(1.25)
    axes[3].imshow(img_dark)
    draw_yolo_boxes(axes[3], main_boxes, W, H)
    axes[3].set_title("4. Giam sang (HSV / Low Light)\nGia lap phong hoc thieu sang / nguoc sang", fontsize=11, fontweight="bold")

    # 5. Tang do sang manh (Den phong hoc chieu truc tiep)
    img_bright = ImageEnhance.Brightness(img).enhance(1.4)
    axes[4].imshow(img_bright)
    draw_yolo_boxes(axes[4], main_boxes, W, H)
    axes[4].set_title("5. Tang do sang (High Exposure)\nGia lap anh sang den phong hoc manh", fontsize=11, fontweight="bold")

    # 6. Mosaic Augmentation (YOLO dac trung)
    mosaic_canvas = Image.new("RGB", (canvas_size, canvas_size), (114, 114, 114))
    half = canvas_size // 2
    sample_imgs = [p for p, b in candidates[:4]]
    while len(sample_imgs) < 4:
        sample_imgs.append(main_img_path)

    offsets = [(0, 0), (half, 0), (0, half), (half, half)]
    for (ox, oy), p_path in zip(offsets, sample_imgs):
        sub_im = Image.open(p_path).convert("RGB").resize((half, half), Image.Resampling.BILINEAR)
        mosaic_canvas.paste(sub_im, (ox, oy))
    axes[5].imshow(mosaic_canvas)
    axes[5].set_title("6. YOLO Mosaic Augmentation\nGhep 4 anh hoc da ngu canh tren 1 frame", fontsize=11, fontweight="bold")

    fig.suptitle("QUY TRINH TIEN XU LY & DATA AUGMENTATION CHO ON-DEVICE AI (320x320)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "augmentation_demo.png", dpi=160)
    plt.close(fig)


def main():
    root = find_root()
    if not root.exists():
        sys.exit(f"Khong tim thay thu muc dataset: {root}")
    out = root.parent / "dataset_report"
    out.mkdir(exist_ok=True)

    results = analyze(root)
    if not results:
        sys.exit("Khong doc duoc tap nao. Kiem tra cau truc images/<split>, labels/<split>.")

    print_report(results)
    save_csv(results, out)
    plot_split(results, out)
    plot_classes(results, out)
    for split in results:
        plot_samples(results, split, out)
    plot_augmentation_demo(results, out)
    print(f"\nDa luu toan bo bao cao va anh minh hoa vao: {out.resolve()}")


if __name__ == "__main__":
    main()
