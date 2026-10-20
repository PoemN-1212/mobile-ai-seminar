package com.example.mobileaiseminar

data class BoundingBox(
    val x1: Float,        // Tọa độ góc trên bên trái (chuẩn hóa 0.0 -> 1.0)
    val y1: Float,
    val x2: Float,        // Tọa độ góc dưới bên phải (chuẩn hóa 0.0 -> 1.0)
    val y2: Float,
    val cx: Float,        // Tọa độ tâm X
    val cy: Float,        // Tọa độ tâm Y
    val w: Float,         // Chiều rộng box
    val h: Float,         // Chiều cao box
    val cnf: Float,       // Độ tin cậy (Confidence score: 0.0 -> 1.0)
    val cls: Int,         // Index lớp (0: person, 1: laptop, 2: cell phone, 3: backpack)
    val clsName: String   // Tên lớp ("person", "laptop", ...)
)