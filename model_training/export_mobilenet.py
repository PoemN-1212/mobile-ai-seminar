import urllib.request
import os

print(">>> Đang tải file mobilenet.tflite chuẩn cho Android...")
# URL tải trực tiếp mô hình SSD MobileNet TFLite chính thức từ ARM ML-Zoo
url = "https://raw.githubusercontent.com/ARM-software/ML-zoo/master/models/object_detection/ssd_mobilenet_v1/tflite_uint8/ssd_mobilenet_v1.tflite"
output_path = "mobilenet.tflite"

urllib.request.urlretrieve(url, output_path)
size_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"✅ ĐÃ TẢI XONG: {output_path} ({size_mb:.2f} MB)")