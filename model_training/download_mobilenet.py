import os
import urllib.request

def download_official_mobilenet():
    # URL trực tiếp từ Google TensorFlow Storage (Chứa đầy đủ metadata và chuẩn 4 output tensors)
    url = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/android/lite-model_ssd_mobilenet_v1_1_metadata_2.tflite"
    output_path = "mobilenet.tflite"

    print(">>> Đang kết nối máy chủ Google Storage...")

    # Thiết lập User-Agent để tránh bị chặn tải tự động
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")]
    urllib.request.install_opener(opener)

    try:
        urllib.request.urlretrieve(url, output_path)
        size_bytes = os.path.getsize(output_path)
        size_mb = size_bytes / (1024 * 1024)

        if size_bytes < 10000:
            print(f"❌ Tải thất bại, file chỉ có {size_bytes} bytes!")
        else:
            print(f"✅ ĐÃ TẢI THÀNH CÔNG: {output_path} ({size_mb:.2f} MB)")
            print(f"📍 Đường dẫn file: {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"❌ Lỗi khi tải file: {e}")

if __name__ == "__main__":
    download_official_mobilenet()