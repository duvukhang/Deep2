import cv2

# --- Ngưỡng AI (Thresholds) ---
EAR_THRESHOLD = 0.18        # Dưới mức này coi là nhắm mắt
MAR_THRESHOLD = 0.5         # Trên mức này coi là ngáp
DROWSY_TIME_STEP = 20       # Số lượng frame liên tiếp nhắm mắt để báo động
POSE_YAW_THRESHOLD = 25     # Độ quay đầu (nhìn sang bên)
POSE_PITCH_THRESHOLD = 15   # Độ gục đầu

# --- Cấu hình Hardware & Camera ---
CAM_ID = 0
IMG_SIZE = (224, 224)
DEVICE = "cuda" # ASUS TUF F15 có RTX, nên dùng cuda

# --- Giao diện (Visualization) ---
FONTS = cv2.FONT_HERSHEY_SIMPLEX
COLORS = {
    "safe": (0, 255, 0),      # Xanh lá
    "warning": (0, 255, 255),  # Vàng
    "danger": (0, 0, 255)      # Đỏ
}

# --- Dịch vụ (Services) ---
OSM_RADIUS = 10000 # Bán kính tìm trạm nghỉ 10km