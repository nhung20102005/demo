from ultralytics import YOLO
import cv2

# ==============================
# CẤU HÌNH
# ==============================
MODEL_PATH = "best (1).pt"
IMAGE_PATH = "image.png"
OUTPUT_PATH = "result.png"

CONFIDENCE = 0.5

# ==============================
# LOAD MODEL YOLO11
# ==============================
print("Đang tải mô hình YOLO11...")

model = YOLO(MODEL_PATH)

print("Các nhãn của mô hình:")
print(model.names)

# ==============================
# NHẬN DIỆN ẢNH
# ==============================
results = model.predict(
    source=IMAGE_PATH,
    conf=CONFIDENCE,
    verbose=False
)

# ==============================
# VẼ KẾT QUẢ
# ==============================
for result in results:

    # Ảnh đã được YOLO vẽ bounding box
    annotated_image = result.plot()

    # Lưu ảnh kết quả
    cv2.imwrite(OUTPUT_PATH, annotated_image)

    # In thông tin nhận diện
    if result.boxes is not None:

        for box in result.boxes:

            # Tọa độ bounding box
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Độ tin cậy
            confidence = float(box.conf[0])

            # ID class
            class_id = int(box.cls[0])

            # Tên class
            class_name = model.names[class_id]

            print(
                f"Phát hiện: {class_name} | "
                f"Confidence: {confidence:.2f} | "
                f"Box: ({int(x1)}, {int(y1)}, "
                f"{int(x2)}, {int(y2)})"
            )

print(f"\nĐã lưu kết quả vào: {OUTPUT_PATH}")

# ==============================
# HIỂN THỊ ẢNH
# ==============================
cv2.imshow("YOLO11 Detection", annotated_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
