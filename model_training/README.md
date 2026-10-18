# BÁO CÁO & MÃ NGUỒN PHẦN 1: TỔNG QUAN AI & QUY TRÌNH HUẤN LUYỆN MÔ HÌNH DI ĐỘNG

> **Dự án Seminar:** Tích hợp On-Device AI với TensorFlow Lite và CameraX trên Android  
> **Phân công trách nhiệm:** **Thành viên 1 - Phần 1 (Lý thuyết nền tảng & Thực nghiệm huấn luyện)**  
> **Căn cứ tài liệu:** `ThuyetMinhAIMobileSeminar.docx`

---

## 📁 Cấu trúc Thư mục Nộp bài

Thư mục đã được làm sạch, giữ lại chính xác các tệp tin và mô hình phục vụ báo cáo và chuyển giao:

```text
Mobile_Present_PJ/
│
├── dataset.yaml                 # File cấu hình YOLO Dataset (đường dẫn & 4 classes)
├── dataset_coco_custom/         # Tập dữ liệu 4 classes (1.45GB - tải tự động qua script hoặc Drive)
│   ├── images/ (train, val, test)
│   └── labels/ (train, val, test)
│
├── dataset_report/              # Thống kê trực quan hóa dữ liệu & Augmentation
│   ├── class_distribution.png  # Biểu đồ phân bố số lượng bounding box 4 lớp
│   ├── split_images.png        # Biểu đồ tỷ lệ phân chia Train/Val/Test
│   ├── augmentation_demo.png   # Trực quan hóa các kỹ thuật Data Augmentation
│   ├── samples_train.png       # Mẫu gán nhãn tập Train
│   ├── samples_val.png         # Mẫu gán nhãn tập Val
│   ├── samples_test.png        # Mẫu gán nhãn tập Test
│   └── stats.csv               # Bảng thống kê chi tiết số lượng bbox
│
├── runs/detect/                 # KẾT QUẢ HUẤN LUYỆN CHÍNH THỨC CỦA CÁC MÔ HÌNH YOLO
│   ├── train_yolov8n/weights/best.pt   # Trọng số TỐT NHẤT sau khi train của YOLOv8n (50 epochs)
│   └── train_yolo11n/weights/best.pt   # Trọng số TỐT NHẤT sau khi train của YOLO11n (50 epochs)
│
├── runs_mobilenet/
│   └── best_mobilenet.pth       # Trọng số TỐT NHẤT SSDLite320-MobileNetV3 (30 epochs)
│
├── benchmark_results/           # Báo cáo đánh giá độc lập trên tập Test (937 ảnh)
│   ├── BENCHMARK_REPORT.md     # Thuyết minh chi tiết kết quả và phân tích nút thắt di động
│   ├── benchmark_summary.csv   # Dữ liệu bảng số liệu benchmark
│   └── charts/
│       ├── map_comparison.png          # Biểu đồ so sánh mAP@50 & mAP@50-95
│       ├── latency_fps_comparison.png  # Biểu đồ độ trễ (ms) và tốc độ suy luận (FPS)
│       ├── model_size_params.png       # Biểu đồ dung lượng file (MB) và số tham số (M)
│       └── per_class_map50.png         # So sánh độ chính xác trên từng class
│
├── download_coco_balanced.py    # Bước 1 & 3: Tải dữ liệu, lọc cân bằng class, chia 70/20/10
├── dataset_report.py            # Bước 2: Tạo báo cáo thống kê & minh họa tăng cường dữ liệu
├── train.py                     # Bước 4 & 5: Huấn luyện YOLOv8n và YOLO11n
├── train_mobilenet.py           # Bước 4 & 5: Huấn luyện SSDLite320-MobileNetV3 (Depthwise Separable)
├── evaluate_models.py           # Bước 6: Đánh giá độc lập trên tập Test & sinh biểu đồ
├── requirements.txt             # Danh sách thư viện cần thiết
├── yolo11n.pt                   # [LƯU Ý] Trọng số gốc ban đầu (Pretrained Base Weight - chưa train)
└── yolov8n.pt                   # [LƯU Ý] Trọng số gốc ban đầu (Pretrained Base Weight - chưa train)
```

> [!IMPORTANT]
> **LƯU Ý QUAN TRỌNG VỀ HAI FILE TRỌNG SỐ Ở THƯ MỤC GỐC (`yolo11n.pt` & `yolov8n.pt`):**
> - Hai file `yolo11n.pt` và `yolov8n.pt` nằm ở thư mục gốc **KHÔNG PHẢI** là mô hình sau khi train của bài toán.
> - Đây là **trọng số khởi tạo ban đầu (Pretrained Base Weights)** do thư viện Ultralytics tự động tải về từ máy chủ chính thức khi khởi chạy `train.py` để nạp các đặc trưng học sâu tổng quát từ 80 lớp COCO gốc làm nền tảng (Backbone) trước khi áp dụng Học chuyển giao (Transfer Learning).
> - **Mô hình chính thức đã huấn luyện xong (50 epochs)** trên 4 classes của bài toán để nộp bài và chuyển giao nằm tại:
>   - **YOLOv8n:** [`runs/detect/train_yolov8n/weights/best.pt`](file:///c:/Study/Mobile_Present_PJ/runs/detect/train_yolov8n/weights/best.pt)
>   - **YOLO11n:** [`runs/detect/train_yolo11n/weights/best.pt`](file:///c:/Study/Mobile_Present_PJ/runs/detect/train_yolo11n/weights/best.pt)
>   - **SSDLite320-MobileNetV3:** [`runs_mobilenet/best_mobilenet.pth`](file:///c:/Study/Mobile_Present_PJ/runs_mobilenet/best_mobilenet.pth)

---

## 🎯 Khớp 6 Bước Huấn Luyện theo Thuyết Minh Seminar

### Bước 1: Thu thập & Gán nhãn dữ liệu (Data Collection & Labeling)
- **Tập dữ liệu:** Lọc từ COCO Dataset 4 lớp đặc trưng cho văn phòng/giảng đường di động:
  1. `person` (người)
  2. `laptop` (máy tính xách tay)
  3. `cell phone` (điện thoại di động)
  4. `backpack` (ba lô)
- **Cân bằng nhãn:** Thu thập cân bằng, định dạng Bounding Box chuẩn hóa theo chuẩn `[class, x_center, y_center, width, height]` (0-1).
- **Mã nguồn:** `download_coco_balanced.py`.

### Bước 2: Tiền xử lý & Tăng cường dữ liệu (Preprocessing & Data Augmentation)
- **Tiền xử lý:** Chuẩn hóa kích thước ảnh về $640 \times 640$ (YOLO) và $320 \times 320$ (MobileNet), chuẩn hóa pixel $[0, 1]$.
- **Data Augmentation:** Tăng cường dữ liệu chống overfitting:
  - Lật ngang ngẫu nhiên (Horizontal Flip, $p=0.5$).
  - Thay đổi độ sáng, độ tương phản (Color Jitter).
  - Biến đổi hình học (Affine, Translation, Scale).
  - Mosaic / Mixup (tích hợp trong pipeline YOLO).
- **Ảnh minh chứng:** Lưu tại `dataset_report/augmentation_demo.png`.

### Bước 3: Phân chia tập dữ liệu (Data Splitting)
- **Train set (70% - 6,558 ảnh):** Cung cấp dữ liệu để mô hình học và cập nhật ma trận trọng số.
- **Validation set (20% - 1,873 ảnh):** Đánh giá sau mỗi epoch để tinh chỉnh siêu tham số và kiểm soát overfitting.
- **Test set (10% - 937 ảnh):** Giữ độc lập hoàn toàn, chỉ nạp vào ở Bước 6 để đánh giá khách quan.

### Bước 4: Lựa chọn kiến trúc mạng tối ưu cho Di động
Chúng ta triển khai và thực nghiệm 3 kiến trúc mobile-first tiêu biểu:
1. **SSDLite320-MobileNetV3-Large:** Ứng dụng **Depthwise Separable Convolution** (tách phép tích chập thành Depthwise Conv và Pointwise $1 \times 1$ Conv), giúp giảm 8–9 lần khối lượng tính toán (FLOPs) so với Standard Convolution.
2. **YOLOv8-Nano (YOLOv8n):** Kiến trúc anchor-free hiện đại từ Ultralytics, thiết kế tối ưu số lượng tham số ($3.14\text{ M}$) cho edge devices.
3. **YOLO11-Nano (YOLO11n):** Kiến trúc thế hệ mới với C3k2 block và SPPF cải tiến, tinh gọn hơn ($2.61\text{ M}$ params).

### Bước 5: Huấn luyện mô hình (Model Training / Fitting)
- **Môi trường:** GPU RTX 3050 Laptop, PyTorch CUDA 12.4, Python 3.13.
- **Quá trình huấn luyện:**
  - `YOLOv8n`: 50 epochs, optimizer AdamW, learning rate scheduling cosine.
  - `YOLO11n`: 50 epochs, optimizer AdamW, learning rate scheduling cosine.
  - `SSDLite320-MobileNetV3`: 30 epochs, optimizer SGD momentum 0.9, cosine annealing LR.
- **Trọng số xuất xưởng:**
  - `runs/detect/train_yolov8n/weights/best.pt`
  - `runs/detect/train_yolo11n/weights/best.pt`
  - `runs_mobilenet/best_mobilenet.pth`

### Bước 6: Đánh giá mô hình & Benchmark độc lập trên tập Test
Kết quả đo đạc thực nghiệm trên tập Test độc lập (937 ảnh):

| Mô hình | Input Size | Số tham số | Dung lượng | mAP@50 | mAP@50-95 | Latency (GPU) | Tốc độ (FPS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO11n** | $640 \times 640$ | **2.61 M** | **5.21 MB** | 46.01% | 29.61% | **2.89 ms** | **346.0** |
| **YOLOv8n** | $640 \times 640$ | 3.14 M | 6.16 MB | **47.19%** | **30.88%** | 3.91 ms | 256.1 |
| **SSDLite320-MobileNetV3** | $320 \times 320$ | 3.76 M | 14.66 MB | 30.31% | 14.52% | 6.13 ms | 163.1 |

---

## 🔄 Đầu Ra Chuyển Giao Cho Thành Viên 2 (Phần 2)

Theo phân công trong `ThuyetMinhAIMobileSeminar.docx`:
1. **Nút thắt kỹ thuật bàn giao:**
   - Các mô hình được huấn luyện bằng PyTorch đang lưu ở định dạng số thực 32-bit (**Float32**).
   - Dung lượng: SSDLite320 ($14.66\text{ MB}$), YOLOv8n ($6.16\text{ MB}$), YOLO11n ($5.21\text{ MB}$).
   - **Vấn đề chuyển giao cho Thành viên 2:** Thiết bị di động không có GPU rời, RAM hạn chế và dễ bị quá nhiệt (Thermal Throttling). Mô hình Float32 cần được chuyển đổi sang định dạng **TensorFlow Lite (.tflite)** kết hợp **Lượng tử hóa (Quantization INT8/FP16)** để tối ưu hóa bộ nhớ và tăng tốc độ xử lý phần cứng NPU/DSP di động.
2. **Tài liệu bàn giao chi tiết:** Đã lập đầy đủ bảng biểu và đồ thị phân tích tại `benchmark_results/BENCHMARK_REPORT.md`.

---

## 📦 QUY TRÌNH BÀN GIAO TẬP DỮ LIỆU (DATASET HANDOVER)

Theo đúng đề cương phân công kỹ thuật của **Bạn 1 (Dữ liệu & Tiền xử lý)** bàn giao cho **Bạn 2 (Thuật toán & Training)**:

> **Lý do kỹ thuật chuẩn ngành:** Thư mục ảnh `dataset_coco_custom/` có dung lượng hơn **1.45 GB (18,740 tệp tin)**. Theo tiêu chuẩn quản trị mã nguồn (Git Best Practices), Git chỉ dùng để quản lý mã nguồn, cấu hình và báo cáo; không commit trực tiếp gigabytes dữ liệu ảnh thô vào Git repository vì sẽ làm nghẽn băng thông, phình to dung lượng repo và vi phạm giới hạn kích thước của GitHub.

### 2 Phương thức bàn giao Dataset cho Thành viên 2 và Thầy cô:

#### 🔹 Cách 1: Tự động tải & tái tạo cục bộ bằng 1 dòng lệnh (Khuyên dùng)
Toàn bộ quy trình thu thập dữ liệu từ COCO API, lọc 4 classes cân bằng và chia tỷ lệ 70/20/10 đã được đóng gói tự động hóa 100%:
```bash
python download_coco_balanced.py
```
*Lệnh này sẽ tự động tạo thư mục `dataset_coco_custom/` với đầy đủ `images/` và `labels/` chuẩn xác 100% khớp với dữ liệu đã huấn luyện.*

#### 🔹 Cách 2: Bàn giao trực tiếp file nén qua Cloud Storage (Google Drive / OneDrive)
1. Thư mục dữ liệu đã xử lý sẵn trên máy: `Mobile_Present_PJ/dataset_coco_custom/`
2. Bạn 1 nén toàn bộ thư mục thành file `dataset_coco_custom.zip` và upload lên thư mục dùng chung của nhóm trên **Google Drive**.
3. Thành viên 2 chỉ cần tải file zip về, giải nén vào thư mục `model_training/` là có thể chạy ngay lệnh huấn luyện hoặc lượng tử hóa TFLite.

