import cv2
from ultralytics import YOLO

# ==========================================
# CẤU HÌNH
# ==========================================

MODEL_PATH = "best (1).pt"

# ID camera:
# 0 = camera mặc định laptop/webcam
# 1 = camera ngoài (nếu có)
CAMERA_ID = 0

# Ngưỡng độ tin cậy
CONFIDENCE = 0.4

# Độ phân giải camera
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720


# ==========================================
# LOAD MODEL
# ==========================================

print("Đang tải mô hình YOLO...")
model = YOLO(MODEL_PATH)

print("Các nhãn của mô hình:")
print(model.names)


# ==========================================
# MỞ CAMERA
# ==========================================

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print(f"Không thể mở camera với ID = {CAMERA_ID}")
    exit()

# Cài đặt độ phân giải camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

print("Đã mở camera thành công.")
print("Nhấn phím Q để thoát.")


# ==========================================
# XỬ LÝ CAMERA REALTIME
# ==========================================

while True:
    ret, frame = cap.read()

    if not ret:
        print("Không đọc được frame từ camera.")
        break

    # ======================================
    # YOLO NHẬN DIỆN
    # ======================================
    results = model.predict(
        source=frame,
        conf=CONFIDENCE,
        verbose=False
    )

    result = results[0]

    # Vẽ khung nhận diện
    annotated_frame = result.plot()

    # ======================================
    # ĐẾM SỐ LƯỢNG LINH KIỆN
    # ======================================
    counts = {}

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        counts[class_name] = counts.get(class_name, 0) + 1

    # ======================================
    # HIỂN THỊ THỐNG KÊ LÊN MÀN HÌNH
    # ======================================
    y = 30

    cv2.putText(
        annotated_frame,
        "LINH KIEN PHAT HIEN:",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    y += 30

    if counts:
        for name, count in counts.items():
            text = f"{name}: {count}"
            cv2.putText(
                annotated_frame,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            y += 30
    else:
        cv2.putText(
            annotated_frame,
            "Khong phat hien",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    # ======================================
    # HIỂN THỊ KẾT QUẢ
    # ======================================
    cv2.imshow("YOLOv11 - Nhan dien linh kien bang camera", annotated_frame)

    # Nhấn q để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# GIẢI PHÓNG TÀI NGUYÊN
# ==========================================

cap.release()
cv2.destroyAllWindows()

print("Đã thoát chương trình.")