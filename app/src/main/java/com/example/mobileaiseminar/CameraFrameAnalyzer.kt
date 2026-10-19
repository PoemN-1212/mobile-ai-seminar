package com.example.mobileaiseminar

import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy

/**
 * CameraFrameAnalyzer - Bộ phân tích khung hình CameraX thời gian thực.
 *
 * Nhiệm vụ kỹ thuật (Thành viên 3 - Gia Bảo):
 * 1. Chặn bắt từng khung hình ImageProxy từ luồng ImageAnalysis.
 * 2. Đọc giá trị góc xoay cảm biến (imageInfo.rotationDegrees).
 * 3. Chuyển đổi buffer nhị phân sang Bitmap chuẩn (RGBA_8888).
 * 4. Xoay ma trận điểm ảnh (Matrix) về đúng chiều hiển thị thẳng đứng của thiết bị.
 * 5. Giải phóng tài nguyên Bitmap cũ và bắt buộc gọi imageProxy.close() để tránh nghẽn luồng.
 */
class CameraFrameAnalyzer(
    private val onFrameProcessed: (bitmap: Bitmap, rotationDegrees: Int) -> Unit
) : ImageAnalysis.Analyzer {

    private val matrix = Matrix()

    override fun analyze(imageProxy: ImageProxy) {
        try {
            // 1. Đọc góc xoay cảm biến của thiết bị thực tế (0, 90, 180, 270 độ)
            val rotationDegrees = imageProxy.imageInfo.rotationDegrees

            // 2. Chuyển đổi ImageProxy sang Bitmap chuẩn
            val rawBitmap = imageProxy.toBitmap()

            // 3. Xoay ma trận điểm ảnh về chiều đứng chuẩn nếu góc xoay khác 0
            val uprightBitmap = if (rotationDegrees != 0) {
                matrix.reset()
                matrix.postRotate(rotationDegrees.toFloat())
                val rotated = Bitmap.createBitmap(
                    rawBitmap, 0, 0,
                    rawBitmap.width, rawBitmap.height,
                    matrix, true
                )
                // Giải phóng rawBitmap trung gian để tránh tràn RAM và giảm áp lực Garbage Collector (GC)
                if (rotated != rawBitmap) {
                    rawBitmap.recycle()
                }
                rotated
            } else {
                rawBitmap
            }

            // 4. Bàn giao Bitmap đã chuẩn hóa sang luồng xử lý suy luận TFLite (Bạn 4)
            onFrameProcessed(uprightBitmap, rotationDegrees)

        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            // 5. BẮT BUỘC PHẢI ĐÓNG: Báo cho CameraX biết frame đã xử lý xong để tiếp nhận frame mới.
            // Nếu quên gọi close(), CameraX sẽ ngừng đẩy frame và ứng dụng bị đông cứng (freeze)!
            imageProxy.close()
        }
    }
}