package com.example.mobileaiseminar

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
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
import com.google.android.material.button.MaterialButtonToggleGroup
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var tvInferenceTime: TextView
    private lateinit var tvFps: TextView
    private lateinit var toggleModel: MaterialButtonToggleGroup

    private lateinit var cameraExecutor: ExecutorService

    // Sử dụng Interface Detector để nạp đa hình cả YOLOv8 và MobileNet
    @Volatile
    private var currentDetector: Detector? = null

    // Cache sẵn 2 instance để gạt nút chuyển tức thì, không bị khựng hình
    private var yoloDetector: YoloDetector? = null
    private var mobileNetDetector: MobileNetDetector? = null

    private var lastFrameTime = 0L
    private var frameCount = 0
    private var currentFps = 0

    companion object {
        private const val CAMERA_PERMISSION_CODE = 1001
        private const val TAG = "MainActivity"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        tvInferenceTime = findViewById(R.id.tvInferenceTime)
        tvFps = findViewById(R.id.tvFps)
        toggleModel = findViewById(R.id.toggleModel)

        cameraExecutor = Executors.newSingleThreadExecutor()

        // 1. Khởi tạo sẵn 2 mô hình trong background
        initDetectors()

        // 2. Bắt sự kiện gạt nút chuyển Model
        setupModelToggle()

        // 3. Khởi chạy Camera
        if (checkCameraPermission()) {
            startCamera()
        } else {
            requestCameraPermission()
        }
    }

    private fun initDetectors() {
        try {
            // Khởi tạo YOLOv8n làm model mặc định ban đầu
            yoloDetector = YoloDetector(this, modelPath = "detect.tflite")
            currentDetector = yoloDetector

            // Khởi tạo MobileNetV3 (sửa tên file đúng với file trong assets)
            mobileNetDetector = MobileNetDetector(this, modelPath = "mobilenet.tflite")
        } catch (e: Exception) {
            Log.e(TAG, "Lỗi nạp mô hình", e)
        }
    }

    private fun setupModelToggle() {
        toggleModel.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (isChecked) {
                when (checkedId) {
                    R.id.btnYolo -> {
                        currentDetector = yoloDetector
                        overlayView.clear()
                        Toast.makeText(this, "Đã chuyển sang: YOLOv8n", Toast.LENGTH_SHORT).show()
                    }
                    R.id.btnMobileNet -> {
                        currentDetector = mobileNetDetector
                        overlayView.clear()
                        Toast.makeText(this, "Đã chuyển sang: MobileNetV3", Toast.LENGTH_SHORT).show()
                    }
                }
            }
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
            Toast.makeText(this, "Yêu cầu cấp quyền Camera để nhận diện!", Toast.LENGTH_LONG).show()
        }
    }

    @SuppressLint("SetTextI18n")
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            imageAnalysis.setAnalyzer(cameraExecutor, CameraFrameAnalyzer { bitmap, _ ->
                val startTime = SystemClock.uptimeMillis()

                // Chạy inference đa hình thông qua interface Detector
                val results = currentDetector?.detect(bitmap) ?: emptyList()

                val inferenceTime = SystemClock.uptimeMillis() - startTime

                // Tính toán FPS thực tế
                frameCount++
                val now = SystemClock.uptimeMillis()
                if (now - lastFrameTime >= 1000) {
                    currentFps = frameCount
                    frameCount = 0
                    lastFrameTime = now
                }

                runOnUiThread {
                    overlayView.setResults(results)
                    tvInferenceTime.text = String.format(Locale.US, "Latency: %d ms", inferenceTime)
                    tvFps.text = String.format(Locale.US, "FPS: %d", currentFps)
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
                Log.e(TAG, "Lỗi gắn CameraX Lifecycle", e)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    override fun onDestroy() {
        super.onDestroy()
        yoloDetector?.close()
        mobileNetDetector?.close()
        cameraExecutor.shutdown()
    }
}