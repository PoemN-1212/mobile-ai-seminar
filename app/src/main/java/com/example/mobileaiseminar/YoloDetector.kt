package com.example.mobileaiseminar

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

class YoloDetector(
    private val context: Context,
    private val modelPath: String = "detect.tflite",
    private val labelPath: String = "labels.txt"
) {
    private var interpreter: Interpreter? = null
    private var labels = mutableListOf<String>()

    private val inputSize = 320 // Kích thước chuẩn mô hình đã train (320x320)
    private val numChannels = 3
    private val numElements = 2100 // Tổng số anchor boxes của YOLOv8 (320x320)
    private val numChannelsOutput = 8 // 4 tọa độ (cx, cy, w, h) + 4 lớp xác suất

    // Ngưỡng lọc
    private val confidenceThreshold = 0.35f // Bỏ các dự đoán dưới 35%
    private val iouThreshold = 0.45f        // Ngưỡng NMS loại bỏ khung trùng lặp

    init {
        loadModel()
        loadLabels()
    }

    private fun loadModel() {
        val options = Interpreter.Options().apply {
            setNumThreads(4) // Tối ưu CPU 4 luồng chạy mượt mà
        }
        val modelBuffer = FileUtil.loadMappedFile(context, modelPath)
        interpreter = Interpreter(modelBuffer, options)
    }

    private fun loadLabels() {
        context.assets.open(labelPath).use { inputStream ->
            BufferedReader(InputStreamReader(inputStream)).useLines { lines ->
                lines.forEach { line ->
                    val trimmed = line.trim()
                    if (trimmed.isNotEmpty()) labels.add(trimmed)
                }
            }
        }
    }

    /**
     * Hàm chính nhận Bitmap từ CameraX và trả về danh sách BoundingBox
     */
    fun detect(bitmap: Bitmap): List<BoundingBox> {
        val tflite = interpreter ?: return emptyList()

        // 1. Tiền xử lý: Resize về 320x320 và đưa vào ByteBuffer chuẩn hóa [0.0, 1.0]
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, false)
        val inputBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * numChannels * 4).apply {
            order(ByteOrder.nativeOrder())
            rewind()
        }

        val intValues = IntArray(inputSize * inputSize)
        resizedBitmap.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize)

        for (pixelValue in intValues) {
            // Chuẩn hóa RGB chia 255.0f
            inputBuffer.putFloat(((pixelValue shr 16) and 0xFF) / 255.0f)
            inputBuffer.putFloat(((pixelValue shr 8) and 0xFF) / 255.0f)
            inputBuffer.putFloat((pixelValue and 0xFF) / 255.0f)
        }

        // 2. Chuẩn bị mảng đầu ra: Ma trận [1, 8, 2100]
        val outputArray = Array(1) { Array(numChannelsOutput) { FloatArray(numElements) } }

        // 3. Chạy suy luận (Inference)
        tflite.run(inputBuffer, outputArray)

        // 4. Giải mã ma trận đầu ra và chạy thuật toán NMS
        return postprocess(outputArray[0])
    }

    private fun postprocess(output: Array<FloatArray>): List<BoundingBox> {
        val boundingBoxes = mutableListOf<BoundingBox>()

        // Quét qua toàn bộ 2100 khung neo dự đoán
        for (i in 0 until numElements) {
            var maxConfidence = -1.0f
            var classIndex = -1

            // 4 dòng đầu là tọa độ, dòng 4 đến 7 là xác suất 4 nhãn
            for (c in 0 until labels.size) {
                val score = output[c + 4][i]
                if (score > maxConfidence) {
                    maxConfidence = score
                    classIndex = c
                }
            }

            // Lọc theo ngưỡng tin cậy
            if (maxConfidence > confidenceThreshold && classIndex != -1) {
                val cx = output[0][i]
                val cy = output[1][i]
                val w = output[2][i]
                val h = output[3][i]

                // Chuyển đổi tọa độ tâm về góc (x1, y1, x2, y2) và chuẩn hóa về dải [0.0 -> 1.0]
                val x1 = max(0f, (cx - w / 2f) / inputSize)
                val y1 = max(0f, (cy - h / 2f) / inputSize)
                val x2 = min(1f, (cx + w / 2f) / inputSize)
                val y2 = min(1f, (cy + h / 2f) / inputSize)

                boundingBoxes.add(
                    BoundingBox(
                        x1 = x1,
                        y1 = y1,
                        x2 = x2,
                        y2 = y2,
                        cx = cx,
                        cy = cy,
                        w = w,
                        h = h,
                        cnf = maxConfidence,
                        cls = classIndex,
                        clsName = labels.getOrElse(classIndex) { "Unknown" }
                    )
                )
            }
        }

        // 5. Áp dụng Non-Maximum Suppression (NMS) để loại bỏ các khung trùng lặp
        return applyNMS(boundingBoxes)
    }

    private fun applyNMS(boxes: List<BoundingBox>): List<BoundingBox> {
        val sortedBoxes = boxes.sortedByDescending { it.cnf }.toMutableList()
        val selectedBoxes = mutableListOf<BoundingBox>()

        while (sortedBoxes.isNotEmpty()) {
            val first = sortedBoxes.removeAt(0)
            selectedBoxes.add(first)

            val iterator = sortedBoxes.iterator()
            while (iterator.hasNext()) {
                val nextBox = iterator.next()
                // Nếu 2 box cùng loại đối tượng và độ đè lấn IoU > ngưỡng -> loại bỏ
                if (first.cls == nextBox.cls && calculateIoU(first, nextBox) > iouThreshold) {
                    iterator.remove()
                }
            }
        }
        return selectedBoxes
    }

    private fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
        val xA = max(box1.x1, box2.x1)
        val yA = max(box1.y1, box2.y1)
        val xB = min(box1.x2, box2.x2)
        val yB = min(box1.y2, box2.y2)

        val intersectionArea = max(0f, xB - xA) * max(0f, yB - yA)
        val box1Area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        val box2Area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)

        val unionArea = box1Area + box2Area - intersectionArea
        return if (unionArea <= 0f) 0f else intersectionArea / unionArea
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }
}