# config.py

class AppConfig:
    HOST = "0.0.0.0"
    PORT = 5000
    CAMERA_INDEX = 0
    DEBUG = False
    ALERT_HISTORY_LIMIT = 30


class DriverConfig:
    # "right": tài xế nằm bên phải khung hình
    # "left" : tài xế nằm bên trái khung hình
    DEFAULT_SIDE = "right"

    # True: chỉ nhận người nằm trong vùng ghế lái
    LOCK_DRIVER_DEFAULT = True

    # ROI mặc định: x1, y1, x2, y2 theo tỉ lệ 0 -> 1
    DEFAULT_ROI_LEFT = {
        "x1": 0.00,
        "y1": 0.08,
        "x2": 0.62,
        "y2": 1.00
    }

    DEFAULT_ROI_RIGHT = {
        "x1": 0.38,
        "y1": 0.08,
        "x2": 1.00,
        "y2": 1.00
    }

    # Mặt người ngồi sau thường nhỏ hơn, dưới ngưỡng này sẽ bị trừ điểm
    SMALL_FACE_RATIO = 0.012
    MEDIUM_FACE_RATIO = 0.020

    # Khi bấm "Khóa tài xế hiện tại", vùng khóa sẽ rộng hơn bbox mặt
    LOCK_ROI_PADDING_X = 1.25
    LOCK_ROI_PADDING_Y = 1.65


class DrowsinessConfig:
    EAR_DEFAULT_THRESHOLD = 0.22
    MAR_THRESHOLD = 0.68

    EYE_CLOSED_SECONDS = 1.8
    YAWN_SECONDS = 1.3

    HEAD_DOWN_THRESHOLD = 0.72

    ALERT_COOLDOWN = 3.0

    LEAN_GRACE_SECONDS = 2.5
    BAD_CAMERA_GRACE_SECONDS = 2.0

    NO_EYE_CONFIDENCE_MIN = 0.25
    NO_EYE_YAWN_SECONDS = 1.5
    NO_EYE_HEAD_DOWN_SECONDS = 2.0
    NO_EYE_DROWSY_SCORE_LIMIT = 7.0

    NORMAL_DROWSY_LIMIT = 6.0
    WARNING_LIMIT = 3.2

    MOUTH_SKIN_RATIO_MIN = 0.18
    SUNGLASSES_DARK_RATIO_MIN = 0.42


class MapConfig:
    DEFAULT_RADIUS = 60000
