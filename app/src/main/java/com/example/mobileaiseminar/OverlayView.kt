package com.example.mobileaiseminar

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    private var results: List<BoundingBox> = emptyList()

    // Cấu hình nét vẽ viền hộp
    private val boxPaint = Paint().apply {
        strokeWidth = 6f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }

    // Cấu hình nền chữ nhãn
    private val textBackgroundPaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    // Cấu hình chữ hiển thị
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 38f
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val textBounds = Rect()

    // Bảng màu riêng cho từng class: 0: Person (Đỏ), 1: Laptop (Xanh lá), 2: Cell phone (Xanh dương), 3: Backpack (Cam)
    private val classColors = listOf(
        Color.rgb(235, 87, 87),   // Person - Đỏ
        Color.rgb(39, 174, 96),   // Laptop - Xanh lá
        Color.rgb(47, 128, 237),  // Cell phone - Xanh dương
        Color.rgb(242, 153, 74)   // Backpack - Cam
    )

    fun setResults(newResults: List<BoundingBox>) {
        results = newResults
        postInvalidate() // Yêu cầu vẽ lại trên Main UI Thread
    }

    fun clear() {
        results = emptyList()
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val viewWidth = width.toFloat()
        val viewHeight = height.toFloat()

        for (box in results) {
            // 1. Ánh xạ tọa độ chuẩn hóa [0.0 -> 1.0] sang tọa độ pixel màn hình
            val left = box.x1 * viewWidth
            val top = box.y1 * viewHeight
            val right = box.x2 * viewWidth
            val bottom = box.y2 * viewHeight

            // Chọn màu theo index của nhãn
            val color = classColors.getOrElse(box.cls) { Color.YELLOW }
            boxPaint.color = color
            textBackgroundPaint.color = color

            // 2. Vẽ viền Bounding Box
            canvas.drawRect(left, top, right, bottom, boxPaint)

            // 3. Chuẩn bị text hiển thị: "laptop 75%"
            val labelText = "${box.clsName} ${(box.cnf * 100).toInt()}%"
            textPaint.getTextBounds(labelText, 0, labelText.length, textBounds)
            val textWidth = textBounds.width().toFloat()
            val textHeight = textBounds.height().toFloat()

            // 4. Vẽ nền màu phía sau chữ
            val bgTop = (top - textHeight - 16f).coerceAtLeast(0f)
            val bgBottom = top.coerceAtLeast(textHeight + 16f)
            val textBgRect = RectF(left, bgTop, left + textWidth + 24f, bgBottom)
            canvas.drawRect(textBgRect, textBackgroundPaint)

            // 5. Vẽ chữ tên nhãn đè lên nền
            canvas.drawText(labelText, left + 12f, bgBottom - 8f, textPaint)
        }
    }
}