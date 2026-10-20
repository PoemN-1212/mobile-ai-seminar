package com.example.mobileaiseminar

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * MainActivity - Điểm khởi động và điều phối CameraX Pipeline.
 *
 * Trách nhiệm kỹ thuật:
 * - Thành viên 3 (Gia Bảo): Quản lý vòng đời CameraX, PreviewView, ImageAnalysis,
 *   chiến lược Backpressure chống tràn RAM và luồng phân tích khung hình.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var tvPerformance: TextView
    private lateinit var cameraExecutor: ExecutorService

    // Biến đo đạc FPS thời gian thực phục vụ báo cáo Seminar
    private var frameCount = 0
    private var lastFpsTimestamp = 0L
    private var currentFps = 0.0

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            startCameraPipeline()
        } else {
            Toast.makeText(
                this,
                "Cần cấp quyền Camera để thực hiện Live Demo nhận diện vật thể!",
                Toast.LENGTH_LONG
            ).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        tvPerformance = findViewById(R.id.tvPerformance)

        // Khởi tạo luồng xử lý ngầm (Background Thread) chuyên biệt cho CameraX
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Kiểm tra và xin quyền Camera theo chuẩn Android Runtime Permissions
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCameraPipeline()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCameraPipeline() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            // 1. USE CASE 1: Hiển thị Preview trực tiếp lên màn hình người dùng
            val previewUseCase = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            // 2. USE CASE 2: Chặn bắt từng frame (ImageAnalysis) để cấp dữ liệu cho AI
            val imageAnalysisUseCase = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                // KỸ THUẬT CỐT LÕI (Thuyết minh Phần 3):
                // STRATEGY_KEEP_ONLY_LATEST: Tự động hủy các frame cũ khi model AI đang bận tính toán,
                // chỉ giữ lại frame mới nhất -> Triệt tiêu hiện tượng giật lag UI và sập Out Of Memory (OOM).
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { analysis ->
                    analysis.setAnalyzer(
                        cameraExecutor,
                        CameraFrameAnalyzer { bitmap, rotationDegrees ->
                            handleInference(bitmap, rotationDegrees)
                        }
                    )
                }

            // Chọn camera sau của điện thoại làm camera chính
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                // Hủy toàn bộ liên kết cũ trước khi gán mới
                cameraProvider.unbindAll()

                // GẮN VÀO VÒNG ĐỜI (Lifecycle-aware):
                // Tự động tắt cảm biến khi app xuống nền (onPause) và bật lại khi mở lại (onResume)
                cameraProvider.bindToLifecycle(
                    this,
                    cameraSelector,
                    previewUseCase,
                    imageAnalysisUseCase
                )
                Log.d("CameraX", "Đã khởi động CameraX thành công (Preview + ImageAnalysis)")
            } catch (exc: Exception) {
                Log.e("CameraX", "Lỗi liên kết CameraX UseCases", exc)
                Toast.makeText(this, "Lỗi khởi động Camera: ${exc.message}", Toast.LENGTH_SHORT).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun handleInference(bitmap: Bitmap, rotationDegrees: Int) {
        // Đo đạc FPS khung hình camera nhận được
        val now = SystemClock.uptimeMillis()
        frameCount++
        if (now - lastFpsTimestamp >= 1000L) {
            currentFps = frameCount * 1000.0 / (now - lastFpsTimestamp)
            frameCount = 0
            lastFpsTimestamp = now

            // Cập nhật thông số lên giao diện Main Thread
            runOnUiThread {
                tvPerformance.text = String.format(
                    "Camera FPS: %.1f | Frame: %dx%d (Xoay: %d°) | Status: Ready for TFLite",
                    currentFps, bitmap.width, bitmap.height, rotationDegrees
                )
            }
        }

        // Bàn giao Bitmap cho Thành viên 4 (Tích hợp TFLite Runtime & Bounding Box)
        // Lưu ý: Sau khi Thành viên 4 xử lý xong, Bitmap có thể tái chế hoặc giữ nguyên theo cấu trúc TFLite
    }

    override fun onDestroy() {
        super.onDestroy()
        // Đóng thread pool khi Activity bị hủy để tránh rò rỉ luồng hệ thống
        cameraExecutor.shutdown()
    }
}