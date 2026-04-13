import cv2
import numpy as np

class HeadPoseEstimator:
    def __init__(self, img_w, img_h):
        self.img_w = img_w
        self.img_h = img_h
        # Tọa độ 3D của các điểm mốc chuẩn trên khuôn mặt (Generic 3D model)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Mũi
            (0.0, -330.0, -65.0),        # Cằm
            (-225.0, 170.0, -135.0),     # Mắt trái
            (225.0, 170.0, -135.0),      # Mắt phải
            (-150.0, -150.0, -125.0),    # Miệng trái
            (150.0, -150.0, -125.0)      # Miệng phải
        ])
        
        # Ma trận Camera (Nội thông số - giả định dựa trên độ phân giải)
        focal_length = img_w
        center = (img_w/2, img_h/2)
        self.camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype = "double"
        )
        self.dist_coeffs = np.zeros((4,1)) # Giả định không có biến dạng ống kính

    def estimate(self, image_points):
        # image_points: Tọa độ 2D từ YOLO-Pose tương ứng với model_points
        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points, image_points, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        # Chuyển đổi Rotation Vector sang Euler Angles (Pitch, Yaw, Roll)
        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.decomposeProjectionMatrix(np.hstack((rmat, translation_vector)))
        
        pitch, yaw, roll = angles.flatten()
        return pitch, yaw, roll