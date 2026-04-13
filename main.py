import cv2
import torch
import numpy as np
import time
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from threading import Thread, Lock
from ultralytics import YOLO # Import YOLO

# Import các module của bạn
from models.hybrid_model import DriverMonitoringSystem
from services.osm_service import OSMService
from configs import settings

app = Flask(__name__)
app.config['SECRET_KEY'] = 'deep2_secret'
socketio = SocketIO(app, async_mode='threading')

# ==========================================
# KHỞI TẠO TÀI NGUYÊN AI (YOLO + HYBRID)
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- Đang khởi động AI trên thiết bị: {device} ---")

# 1. Load Mắt thần (YOLO)
try:
    yolo_model = YOLO('weights/yolo11n.pt') 
    print("✅ Đã load YOLO thành công!")
except Exception as e:
    print(f"❌ Lỗi load YOLO: {e}")

# 2. Load Não bộ (Hybrid Model)
try:
    hybrid_brain = DriverMonitoringSystem(feature_dim=256).to(device)
    hybrid_brain.load_state_dict(torch.load("weights/hybrid_final.pth", map_location=device))
    hybrid_brain.eval()
    print("✅ Đã load Não bộ Hybrid thành công!")
except Exception as e:
    print(f"❌ Lỗi load Hybrid Model: {e}")

osm = OSMService()
lock = Lock()
output_frame = None

# ==========================================
# LUỒNG AI CHÍNH (THỰC CHIẾN 100%)
# ==========================================
def ai_detection_thread():
    global output_frame
    cap = cv2.VideoCapture(0) # Camera Laptop
    
    # Bộ đệm lưu trữ 30 frames liên tiếp cho mô hình Transformer
    seq_length = 30
    spatial_buffer = []
    geo_buffer = []
    
    while True:
        success, frame = cap.read()
        if not success: 
            time.sleep(0.1)
            continue
        
        # 1. YOLO Inference (Nhận diện vật thể)
        results = yolo_model(frame, verbose=False)[0]
        
        # --- BẢN VÁ LỖI: LOGIC MẮT ƯU TIÊN ---
        mar = 0.1 
        face_box = None
        eye_open_detected = False
        eye_closed_detected = False
        
        # 2. Phân tích kết quả từ YOLO
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Lọc nhiễu: Chỉ lấy các nhận diện chắc chắn > 40%
            if conf < 0.4:
                continue
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            
            if cls_id == 0: eye_closed_detected = True
            elif cls_id == 1: eye_open_detected = True
            elif cls_id == 2: face_box = (x1, y1, x2, y2)
            elif cls_id == 3: mar = 0.8
                
        # Cập nhật EAR chính xác
        if eye_open_detected: ear = 0.35
        elif eye_closed_detected: ear = 0.10
        else: ear = 0.30
        # ------------------------------------
        
        # Giả lập Head Pose 
        pitch, yaw, roll = 0.0, 0.0, 0.0 
        
        geo_vector = [ear, mar, pitch, yaw, roll, 1.0]
        spatial_vector = np.zeros(512) 
        
        geo_buffer.append(geo_vector)
        spatial_buffer.append(spatial_vector)
        
        if len(geo_buffer) > seq_length:
            geo_buffer.pop(0)
            spatial_buffer.pop(0)
            
        # 3. Kích hoạt Não bộ (Khi đủ 30 frames)
        is_drowsy = False
        drowsy_prob = 0.0
        
        if len(geo_buffer) == seq_length:
            with torch.no_grad():
                geo_tensor = torch.tensor([geo_buffer], dtype=torch.float32).to(device)
                spatial_tensor = torch.tensor([spatial_buffer], dtype=torch.float32).to(device)
                
                outputs = hybrid_brain(spatial_tensor, geo_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                drowsy_prob = probabilities[1].item()
                
                # --- BẢN VÁ LỖI: ĐẢM BẢO BÁO ĐỘNG KHI NHẮM MẮT ---
                # Tính trung bình EAR của 15 frame gần nhất (~0.5 giây)
                recent_ears = [frame_data[0] for frame_data in geo_buffer[-15:]]
                avg_ear = sum(recent_ears) / len(recent_ears)
                
                if avg_ear <= 0.15:
                    is_drowsy = True
                    drowsy_prob = 0.99 # Ép cảnh báo lên 99%
                else:
                    is_drowsy = drowsy_prob > 0.7 
                # -------------------------------------------------
        
        # 4. Gửi dữ liệu lên Web Dashboard
        socketio.emit('update_data', {
            'ear': round(ear, 2),
            'pose_status': "Bình thường",
            'is_drowsy': bool(is_drowsy),
            'occlusion': "Không", # --- BẢN VÁ LỖI: Hết bị 'undefined' trên Web ---
            'prob': f"{drowsy_prob*100:.1f}%"
        })
        
        # 5. Vẽ cảnh báo lên màn hình
        cv2.putText(frame, f"EAR: {ear:.2f} | Ti le ngu gat: {drowsy_prob*100:.0f}%", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                    (0, 0, 255) if is_drowsy else (0, 255, 0), 2)
        
        if is_drowsy:
            cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (0,0,255), 8)
            cv2.putText(frame, "CANH BAO NGU GAT!", (frame.shape[1]//2 - 150, frame.shape[0]//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        with lock:
            output_frame = frame.copy()

# ==========================================
# CÁC ROUTE CỦA WEB
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

def generate():
    global output_frame, lock
    while True:
        if output_frame is None:
            time.sleep(0.05)
            continue
        with lock:
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
        if not flag:
            continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('find_stops_request')
def handle_osm_request(data):
    stops = osm.find_nearest_rest_stop(data['lat'], data['lon'])
    socketio.emit('rest_stops_data', {'stops': stops})

if __name__ == '__main__':
    t = Thread(target=ai_detection_thread)
    t.daemon = True
    t.start()
    print("--- Khởi động Web Server tại http://127.0.0.1:5000 ---")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)