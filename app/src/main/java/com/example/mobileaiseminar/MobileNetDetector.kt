package com.example.mobileaiseminar

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MobileNetDetector(
    private val context: Context,
    private val modelPath: String = "mobilenet.tflite"
) : Detector {

    private var interpreter: Interpreter? = null
    private val inputSize = 300 // Chuẩn input của SSD-MobileNet
    private val maxDetections = 10
    private val confidenceThreshold = 0.40f

    // Ánh xạ index gốc COCO sang 4 class của nhóm:
    // 1: Person -> 0 (Đỏ), 73: Laptop -> 1 (Xanh lá), 77: Cell phone -> 2 (Xanh dương), 27: Backpack -> 3 (Cam)
    private val cocoTargetMap = mapOf(
        1 to Pair(0, "person"),
        73 to Pair(1, "laptop"),
        77 to Pair(2, "cell phone"),
        27 to Pair(3, "backpack")
    )

    init {
        loadModel()
    }

    private fun loadModel() {
        try {
            val options = Interpreter.Options().apply { setNumThreads(4) }
            val modelBuffer = FileUtil.loadMappedFile(context, modelPath)
            interpreter = Interpreter(modelBuffer, options)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    override fun detect(bitmap: Bitmap): List<BoundingBox> {
        val tflite = interpreter ?: return emptyList()

        val inputTensor = tflite.getInputTensor(0)
        val isQuantized = inputTensor.dataType() == DataType.UINT8

        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, false)
        val bytesPerChannel = if (isQuantized) 1 else 4
        val inputBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * bytesPerChannel).apply {
            order(ByteOrder.nativeOrder())
            rewind()
        }

        val intValues = IntArray(inputSize * inputSize)
        resizedBitmap.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize)

        for (pixel in intValues) {
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF

            if (isQuantized) {
                inputBuffer.put(r.toByte())
                inputBuffer.put(g.toByte())
                inputBuffer.put(b.toByte())
            } else {
                inputBuffer.putFloat((r - 127.5f) / 127.5f)
                inputBuffer.putFloat((g - 127.5f) / 127.5f)
                inputBuffer.putFloat((b - 127.5f) / 127.5f)
            }
        }

        // 4 mảng output chuẩn của SSD-MobileNet
        val outputBoxes = Array(1) { Array(maxDetections) { FloatArray(4) } } // [ymin, xmin, ymax, xmax]
        val outputClasses = Array(1) { FloatArray(maxDetections) }
        val outputScores = Array(1) { FloatArray(maxDetections) }
        val numDetections = FloatArray(1)

        val outputs = mapOf(
            0 to outputBoxes,
            1 to outputClasses,
            2 to outputScores,
            3 to numDetections
        )

        tflite.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)

        val results = mutableListOf<BoundingBox>()
        val count = numDetections[0].toInt().coerceAtMost(maxDetections)

        for (i in 0 until count) {
            val score = outputScores[0][i]
            if (score >= confidenceThreshold) {
                val rawClassId = outputClasses[0][i].toInt() + 1 // COCO index

                // Chỉ lấy các đối tượng thuộc 4 lớp mục tiêu
                val targetInfo = cocoTargetMap[rawClassId]
                if (targetInfo != null) {
                    val ymin = outputBoxes[0][i][0].coerceIn(0f, 1f)
                    val xmin = outputBoxes[0][i][1].coerceIn(0f, 1f)
                    val ymax = outputBoxes[0][i][2].coerceIn(0f, 1f)
                    val xmax = outputBoxes[0][i][3].coerceIn(0f, 1f)

                    results.add(
                        BoundingBox(
                            x1 = xmin,
                            y1 = ymin,
                            x2 = xmax,
                            y2 = ymax,
                            cx = (xmin + xmax) / 2f,
                            cy = (ymin + ymax) / 2f,
                            w = xmax - xmin,
                            h = ymax - ymin,
                            cnf = score,
                            cls = targetInfo.first,
                            clsName = targetInfo.second
                        )
                    )
                }
            }
        }
        return results
    }

    override fun close() {
        interpreter?.close()
        interpreter = null
    }
}