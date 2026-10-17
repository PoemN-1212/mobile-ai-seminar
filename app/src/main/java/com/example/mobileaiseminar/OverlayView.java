package com.example.mobileaiseminar;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;
import java.util.ArrayList;
import java.util.List;

public class OverlayView extends View {
    private final Paint boxPaint = new Paint();
    private final Paint textPaint = new Paint();
    private List<DetectionResult> results = new ArrayList<>();
    // Tạo sẵn một đối tượng RectF dùng chung
    private final RectF rect = new RectF();
    public OverlayView(android.content.Context context) {
        super(context);
    }
    private void initPaints() {
        // Cấu hình viền của Bounding Box
        boxPaint.setColor(Color.GREEN);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(8f); // Độ dày nét vẽ

        // Cấu hình chữ hiển thị tên vật thể & % tin cậy
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(45f);
        textPaint.setStyle(Paint.Style.FILL);
        textPaint.setShadowLayer(5.0f, 0f, 0f, Color.BLACK); // Đổ bóng đen để chữ không bị chìm
    }

    // Bạn 4 (phụ trách TFLite) sẽ gọi hàm này và truyền danh sách kết quả vào đây
    public void setResults(List<DetectionResult> newResults) {
        this.results = newResults;
        invalidate(); // Lệnh này yêu cầu Android lập tức gọi lại hàm onDraw() để vẽ khung mới
    }

    @Override
    protected void onDraw(android.graphics.Canvas canvas) {
        super.onDraw(canvas);
        if (results == null || results.isEmpty()) return;

        int screenWidth = getWidth();
        int screenHeight = getHeight();
        for (DetectionResult result : results) {
            // Thuật toán Mapping: Scale tọa độ tương đối (0.0 -> 1.0) sang kích thước pixel thực tế của máy
            float left = result.box[0] * screenWidth;
            float top = result.box[1] * screenHeight;
            float right = result.box[2] * screenWidth;
            float bottom = result.box[3] * screenHeight;

            rect.set(left, top, right, bottom);

            // Vẽ khung chữ nhật đè lên camera
            canvas.drawRect(rect, boxPaint);

            // Vẽ text tên nhãn và điểm tin cậy (%)
            String labelText = result.label + " " + Math.round(result.score * 100) + "%";
            canvas.drawText(labelText, left, top - 15, textPaint);
        }
    }
    // Class cấu trúc dữ liệu chuẩn để Bạn 4 truyền sang cho Bạn
    public static class DetectionResult {
        public float[] box; // Mảng 4 phần tử [trái, trên, phải, dưới] dạng 0.0 -> 1.0
        public String label; // Tên vật thể (Person, Laptop...)
        public float score; // Độ tin cậy (VD: 0.95)

        public DetectionResult(float[] box, String label, float score) {
            this.box = box;
            this.label = label;
            this.score = score;
        }
    }
}
