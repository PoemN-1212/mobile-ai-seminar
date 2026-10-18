from ultralytics import YOLO

def train_pipeline():
    # 1. Đường dẫn file cấu hình bộ dữ liệu đã cân bằng
    DATASET_CONFIG = "dataset_coco_custom/dataset.yaml"
    
    # 2. Cấu hình tối ưu riêng cho Card RTX 3050 (4GB VRAM)
    EPOCHS = 50          # 50 epoch là chuẩn đẹp để đạt mAP cao
    IMG_SIZE = 320       # Chuẩn kích thước Mobile (theo đề tài Live Demo)
    BATCH_SIZE = 16      # 16 là mức tối ưu nhất cho 4GB VRAM ở kích thước 320x320
    DEVICE = 0           # Chạy trên GPU 0 (RTX 3050)
    WORKERS = 2          # 2 worker giúp nạp ảnh ổn định trên Windows

    # =======================================================
    # GIAI ĐOẠN 1: HUẤN LUYỆN MODEL CHÍNH (YOLOv8n)
    # =======================================================
    print("\n" + "="*60)
    print(">>> [1/2] BẮT ĐẦU HUẤN LUYỆN MODEL CHÍNH: YOLOv8n")
    print("="*60)
    
    # Tự động tải weights gốc pretrained từ Ultralytics
    model_v8 = YOLO("yolov8n.pt")
    
    model_v8.train(
        data=DATASET_CONFIG,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        name="train_yolov8n",    # Thư mục lưu kết quả
        plots=True
    )

    # =======================================================
    # GIAI ĐOẠN 2: HUẤN LUYỆN MODEL ĐỐI CHỨNG (YOLO11n)
    # =======================================================
    print("\n" + "="*60)
    print(">>> [2/2] BẮT ĐẦU HUẤN LUYỆN MODEL ĐỐI CHỨNG: YOLO11n")
    print("="*60)
    
    model_11 = YOLO("yolo11n.pt")
    
    model_11.train(
        data=DATASET_CONFIG,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        name="train_yolo11n",    # Thư mục lưu kết quả đối chứng
        plots=True
    )

    print("\n✅ ĐÃ HUẤN LUYỆN XONG CẢ 2 MODEL!")

if __name__ == "__main__":
    train_pipeline()
