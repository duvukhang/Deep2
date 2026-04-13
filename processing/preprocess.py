import cv2
import numpy as np

def apply_clahe(image):
    """Cân bằng độ sáng cục bộ (Hữu ích cho lái xe ban đêm)"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def detect_occlusion(yolo_results):
    """
    Logic xác định vật cản dựa trên class của YOLO
    Nếu thấy nhãn 'mask' hoặc 'sunglasses' với confidence cao
    """
    occlusion_status = {"mask": False, "sunglasses": False}
    for r in yolo_results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = r.names[cls]
            if label == 'mask': occlusion_status['mask'] = True
            if label == 'sunglasses': occlusion_status['sunglasses'] = True
    return occlusion_status