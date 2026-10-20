package com.example.mobileaiseminar

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.SystemClock
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var tvInferenceTime: TextView
    private lateinit var tvFps: TextView

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var detector: YoloDetector

    private val CAMERA_PERMISSION_CODE = 1001

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        tvInferenceTime = findViewById(R.id.tvInferenceTime)
        tvFps = findViewById(R.id.tvFps)

        // 1. Khởi tạo bộ suy luận YOLOv8n TFLite
        detector = YoloDetector(this)

        // 2. Tạo luồng chạy nền riêng biệt cho AI để không gây lag giao diện
        cameraExecutor = Executors.newSingleThreadExecutor()

        // 3. Kiểm tra và yêu cầu quyền Camera
        if (checkCameraPermission()) {
            startCamera()
        } else {
            requestCameraPermission()
        }
    }

    private fun checkCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestCameraPermission() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.CAMERA),
            CAMERA_PERMISSION_CODE
        )
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_CODE && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            Toast.makeText(this, "Ứng dụng cần quyền Camera để nhận diện!", Toast.LENGTH_LONG).show()
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            // Cấu hình UseCase hiển thị Preview
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            // Cấu hình UseCase phân tích hình ảnh (ImageAnalysis)
            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            // Gắn Analyzer đã tối ưu bộ nhớ
            imageAnalysis.setAnalyzer(cameraExecutor, CameraFrameAnalyzer { bitmap, _ ->
                val startTime = SystemClock.uptimeMillis()

                // Chạy suy luận TFLite
                val results = detector.detect(bitmap)

                val inferenceTime = SystemClock.uptimeMillis() - startTime
                val fps = if (inferenceTime > 0) 1000 / inferenceTime else 0

                // Cập nhật kết quả lên Main UI Thread
                runOnUiThread {
                    overlayView.setResults(results)
                    tvInferenceTime.text = "Inference: $inferenceTime ms"
                    tvFps.text = "FPS: $fps"
                }
            })

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    cameraSelector,
                    preview,
                    imageAnalysis
                )
            } catch (e: Exception) {
                e.printStackTrace()
            }

        }, ContextCompat.getMainExecutor(this))
    }

    override fun onDestroy() {
        super.onDestroy()
        detector.close()
        cameraExecutor.shutdown()
    }
}