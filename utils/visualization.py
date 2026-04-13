import cv2
import numpy as np

def draw_info(frame, ear, pose, status, prob):
    h, w, _ = frame.shape
    
    # Vẽ nền mờ cho text
    cv2.rectangle(frame, (0, 0), (250, 150), (0, 0, 0), -1)
    cv2.addWeighted(frame, 0.5, frame, 0.5, 0)

    # Hiển thị chỉ số
    color = (0, 255, 0) if status == "AWAKE" else (0, 0, 255)
    cv2.putText(frame, f"STATUS: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Prob: {prob:.2%}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Vẽ vector hướng đầu (Visualizing Pose)
    # Tưởng tượng vẽ một đường thẳng từ mũi tài xế ra ngoài
    pass