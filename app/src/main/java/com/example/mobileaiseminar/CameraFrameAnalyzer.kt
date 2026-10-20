package com.example.mobileaiseminar

import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy

/**
 * CameraFrameAnalyzer - Bộ phân tích khung hình CameraX thời gian thực tối ưu bộ nhớ.
 *
 * Đã tối ưu:
 * 1. Đọc và chuẩn hóa góc xoay cảm biến (rotationDegrees).
 * 2. Cơ chế khóa luồng (isProcessing) chống dồn frame khi AI chạy chậm hơn Camera.
 * 3. Bắt toàn diện Throwable (chống văng app do OutOfMemoryError).
 * 4. Tự động thu hồi Bitmap sau khi xử lý xong để giải phóng RAM tức thì.
 */
class CameraFrameAnalyzer(
    private val onFrameProcessed: (bitmap: Bitmap, frameTimeMs: Long) -> Unit
) : ImageAnalysis.Analyzer {

    private val matrix = Matrix()

    // Cờ kiểm soát: chỉ cho phép lấy frame tiếp theo khi frame hiện tại đã suy luận xong
    @Volatile
    private var isProcessing = false

    override fun analyze(imageProxy: ImageProxy) {
        // Nếu AI đang bận xử lý frame trước đó, bỏ qua frame hiện tại ngay lập tức
        if (isProcessing) {
            imageProxy.close()
            return
        }

        val startTime = SystemClock.uptimeMillis()

        try {
            isProcessing = true

            // 1. Đọc góc xoay từ cảm biến thiết bị
            val rotationDegrees = imageProxy.imageInfo.rotationDegrees

            // 2. Chuyển đổi ImageProxy sang Bitmap (CameraX 1.3+ native)
            val rawBitmap = imageProxy.toBitmap()

            // 3. Xoay ma trận điểm ảnh về chiều đứng chuẩn (Portrait)
            val uprightBitmap = if (rotationDegrees != 0) {
                matrix.reset()
                matrix.postRotate(rotationDegrees.toFloat())
                val rotated = Bitmap.createBitmap(
                    rawBitmap, 0, 0,
                    rawBitmap.width, rawBitmap.height,
                    matrix, true
                )

                // Giải phóng ngay bitmap thô trung gian
                if (rotated != rawBitmap) {
                    rawBitmap.recycle()
                }
                rotated
            } else {
                rawBitmap
            }

            val frameTimeMs = SystemClock.uptimeMillis() - startTime

            // 4. Bàn giao Bitmap thẳng đứng cho bộ suy luận YoloDetector
            onFrameProcessed(uprightBitmap, frameTimeMs)

            // 5. Giải phóng uprightBitmap ngay sau khi dùng xong để tránh tràn RAM
            uprightBitmap.recycle()

        } catch (t: Throwable) {
            // Bắt cả Exception và OutOfMemoryError để bảo vệ app không bao giờ bị văng
            t.printStackTrace()
        } finally {
            // Mở khóa cờ và giải phóng frame camera
            isProcessing = false
            imageProxy.close()
        }
    }
}