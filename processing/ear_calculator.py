import numpy as np
from scipy.spatial import distance as dist

def calculate_ear(eye_landmarks):
    # eye_landmarks: list of (x, y) coordinates
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

def get_head_pose(landmarks, img_w, img_h):
    # Sử dụng SolvePnP để tính Pitch, Yaw, Roll
    # Trả về vector 3 chiều đại diện cho hướng nhìn của đầu
    # Đây là chìa khóa để xử lý khi tài xế đeo kính râm
    pass