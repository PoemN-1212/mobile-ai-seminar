"""
export_tflite_quantization.py
Chuyen doi va Luong tu hoa mo hinh Object Detection sang TensorFlow Lite (TFLite INT8 Quantization)
phuc vu tich hop tren thiet bi Android (CameraX + TFLite Runtime).

Phan cong trach nhiem: Gia Bao - Phan 3 (CameraX + TFLite Converter & Quantization)
"""

import os
import shutil
import sys
from pathlib import Path

# Dam bao hien thi ky tu Unicode tieng Viet tren Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 4 lop doi tuong theo chuan dataset.yaml
CLASSES = ["person", "laptop", "cell phone", "backpack"]
ANDROID_ASSETS_DIR = Path("../app/src/main/assets")
WEIGHTS_DIR = Path("runs/detect/train_yolov8n/weights")
SAVED_MODEL_DIR = WEIGHTS_DIR / "yolov8n_saved_model"
BEST_ONNX = WEIGHTS_DIR / "best.onnx"
BEST_PT = WEIGHTS_DIR / "best.pt"


def ensure_saved_model():
    """Kiem tra va tao SavedModel tu best.onnx neu chua co."""
    if SAVED_MODEL_DIR.exists() and (SAVED_MODEL_DIR / "saved_model.pb").exists():
        print(f"[+] Da tim thay SavedModel hop le tai: {SAVED_MODEL_DIR.resolve()}")
        return SAVED_MODEL_DIR

    print("[*] Bat dau chuyen doi mo hinh sang TensorFlow SavedModel...")
    
    # 1. Kiem tra file ONNX, neu chua co thi xuat tu PyTorch best.pt
    if not BEST_ONNX.exists():
        if not BEST_PT.exists():
            raise FileNotFoundError(f"Khong tim thay file trong so: {BEST_PT}")
        print(f"[*] Xuat ONNX tu: {BEST_PT}")
        from ultralytics import YOLO
        model = YOLO(str(BEST_PT))
        model.export(format="onnx", imgsz=320, opset=12)

    # 2. Dung onnx2tf de bien doi sang SavedModel
    print(f"[*] Chuyen doi ONNX ({BEST_ONNX}) sang TensorFlow SavedModel...")
    import onnx2tf
    onnx2tf.convert(
        input_onnx_file_path=str(BEST_ONNX),
        output_folder_path=str(SAVED_MODEL_DIR),
        non_verbose=True,
    )
    print(f"[+] Chuyen doi thanh cong SavedModel vao: {SAVED_MODEL_DIR.resolve()}")
    return SAVED_MODEL_DIR


def quantize_and_export():
    print("=" * 70)
    print("  QUY TRINH LUONG TU HOA TENSORFLOW LITE (TFLITE INT8 QUANTIZATION)")
    print("  Seminar: Tich hop On-Device AI voi TFLite va CameraX tren Android")
    print("=" * 70)

    # 1. Tao file nhan labels.txt
    ANDROID_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    labels_content = "\n".join(CLASSES) + "\n"
    
    target_labels = ANDROID_ASSETS_DIR / "labels.txt"
    target_labels.write_text(labels_content, encoding="utf-8")
    Path("labels.txt").write_text(labels_content, encoding="utf-8")
    print(f"[+] Da tao file labels.txt: {len(CLASSES)} classes -> {target_labels.resolve()}")

    # 2. Chuan bi SavedModel
    saved_model_path = ensure_saved_model()

    # 3. Import TensorFlow & Khoi tao TFLite Converter
    import tensorflow as tf
    print("\n[*] Dang cau hinh TensorFlow Lite Converter...")
    print(f"    TensorFlow version: {tf.__version__}")

    # --- A. Export Float32 Model (Chuan goc lam moc so sanh) ---
    converter_f32 = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
    tflite_f32 = converter_f32.convert()
    f32_path = Path("detect_float32.tflite")
    f32_path.write_bytes(tflite_f32)
    f32_size_mb = len(tflite_f32) / (1024 * 1024)
    print(f"[+] 1. Mo hinh goc Float32: {f32_path.resolve()} ({f32_size_mb:.2f} MB)")

    # --- B. Export Float16 Model (Nen ban chinh xac 16-bit) ---
    converter_f16 = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
    converter_f16.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_f16.target_spec.supported_types = [tf.float16]
    tflite_f16 = converter_f16.convert()
    f16_path = Path("detect_float16.tflite")
    f16_path.write_bytes(tflite_f16)
    f16_size_mb = len(tflite_f16) / (1024 * 1024)
    print(f"[+] 2. Mo hinh Float16 (FP16): {f16_path.resolve()} ({f16_size_mb:.2f} MB)")

    # --- C. Export INT8 Dynamic Range Quantization Model (Toi uu Mobile) ---
    print("[*] Dang thuc hien Luong tu hoa INT8 (Dynamic Range Quantization)...")
    converter_int8 = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_int8 = converter_int8.convert()
    int8_path = Path("detect_int8.tflite")
    int8_path.write_bytes(tflite_int8)
    int8_size_mb = len(tflite_int8) / (1024 * 1024)
    reduction_pct = ((f32_size_mb - int8_size_mb) / f32_size_mb) * 100.0
    print(f"[+] 3. Mo hinh Luong tu hoa INT8: {int8_path.resolve()} ({int8_size_mb:.2f} MB)")
    print(f"    => Giam {reduction_pct:.1f}% dung luong so voi Float32 goc!")

    # 4. Sao chep model INT8 chinh thuc vao Android assets (detect.tflite)
    target_tflite = ANDROID_ASSETS_DIR / "detect.tflite"
    shutil.copy2(int8_path, target_tflite)
    shutil.copy2(int8_path, Path("detect.tflite"))
    print(f"\n[+] DA NOP MO HINH VAO ANDROID ASSETS:")
    print(f"    -> Model: {target_tflite.resolve()} ({target_tflite.stat().st_size / (1024*1024):.2f} MB)")
    print(f"    -> Labels: {target_labels.resolve()}")

    # 5. Kiem tra tinh hop le bang TFLite Interpreter (Kiem dinh ky thuat)
    print("\n[*] Kiem tra dac tinh Tensor cua mo hinh sau khi luong tu hoa:")
    interpreter = tf.lite.Interpreter(model_path=str(target_tflite))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_shape = input_details[0]["shape"]
    in_dtype = input_details[0]["dtype"].__name__
    out_shape = output_details[0]["shape"]
    out_dtype = output_details[0]["dtype"].__name__

    print(f"    - Input  Tensor: Name='{input_details[0]['name']}', Shape={list(in_shape)}, Dtype={in_dtype}")
    print(f"    - Output Tensor: Name='{output_details[0]['name']}', Shape={list(out_shape)}, Dtype={out_dtype}")

    print("\n" + "=" * 70)
    print("  BANG TONG HOP SO LIEU THUYET MINH SEMINAR (PHAN 3 & PHAN 4)")
    print("=" * 70)
    print(f"  * Kien truc mang goc:        YOLOv8-Nano (Anchor-free Mobile Architecture)")
    print(f"  * Kich thuoc Tensor dau vao: {in_shape[1]}x{in_shape[2]}x{in_shape[3]} (RGB Bitmap)")
    print(f"  * Cac lop nhan dien (4):     {', '.join(CLASSES)}")
    print(f"  * Dung luong Float32 goc:    {f32_size_mb:.2f} MB (100%)")
    print(f"  * Dung luong Float16 (FP16): {f16_size_mb:.2f} MB ({f16_size_mb/f32_size_mb*100:.1f}%)")
    print(f"  * Dung luong INT8 Quantized: {int8_size_mb:.2f} MB ({int8_size_mb/f32_size_mb*100:.1f}%) -> Giam {reduction_pct:.1f}%")
    print(f"  * Output Box candidates:     {out_shape[2]} candidates x {out_shape[1]} channels (4 bbox + 4 classes)")
    print("=" * 70)
    print("[SUCCESS] Hoan tat toan bo quy trinh xuat va luong tu hoa TFLite!")


if __name__ == "__main__":
    quantize_and_export()
    quantize_and_export()