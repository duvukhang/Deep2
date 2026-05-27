import cv2
import torch
import numpy as np
import time
import requests # Thêm thư viện này để gọi API lấy IP
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from threading import Thread, Lock
from ultralytics import YOLO

# Import các module của bạn
from models.hybrid_model import DriverMonitoringSystem
from services.osm_service import OSMService
from configs import settings

app = Flask(__name__)
app.config['SECRET_KEY'] = 'deep2_secret'
socketio = SocketIO(app, async_mode='threading')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- Đang khởi động AI trên thiết bị: {device} ---")

# 1. Load Mắt thần (YOLO)
try:
    yolo_model = YOLO('runs/detect/runs/detect/yolo_drowsy_v1/weights/best.pt')
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

def get_location_by_ip():
    """Lấy tọa độ dựa trên IP public của máy tính."""
    try:
        response = requests.get('http://ip-api.com/json/', timeout=3).json()
        return response['lat'], response['lon']
    except Exception as e:
        print(f"Lỗi lấy IP: {e}. Đang dùng tọa độ mặc định (TP.HCM).")
        return 10.762622, 106.660172 

def ai_detection_thread():
    global output_frame
    
    # ==========================================
    # FIX LỖI CAMERA ĐEN THUI TRÊN WINDOWS
    # Thêm cv2.CAP_DSHOW để dùng DirectShow thay vì MSMF
    # ==========================================
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
    
    seq_length = 30
    spatial_buffer = []
    geo_buffer = []
    
    # --- CÁC BIẾN THEO DÕI TRẠNG THÁI (STATE TRACKING) ---
    missing_face_frames = 0
    normal_y_center = None
    has_suggested_rest = False
    awake_frames_count = 0 
    
    while True:
        success, frame = cap.read()
        if not success: 
            time.sleep(0.1)
            continue
        
        results = yolo_model(frame, verbose=False)[0]
        
        ear = 0.35 
        mar = 0.1 
        driver_box = None
        eye_open_detected = False
        eye_closed_detected = False
        is_distracted = False
        
        # 1. Lọc và tìm BBox của Tài xế (Khuôn mặt to nhất / Gần cam nhất)
        faces = [box for box in results.boxes if int(box.cls[0]) == 2 and float(box.conf[0]) >= 0.4]
        if faces:
            faces_sorted = sorted(faces, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]), reverse=True)
            driver_box = faces_sorted[0]

        # 2. Xử lý Mất tập trung / Cúi người / Lấy đồ
        if driver_box is None:
            missing_face_frames += 1
            if missing_face_frames > 15: # Mất mặt khoảng 0.5s
                cv2.putText(frame, "TAI XE KHUAT TAM NHIN!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
                geo_buffer.clear()
                spatial_buffer.clear()
                is_distracted = True
                ear = 0.35 # Ép tỉnh táo để không cộng dồn điểm ngủ gật
        else:
            missing_face_frames = 0
            dx1, dy1, dx2, dy2 = map(int, driver_box.xyxy[0])
            current_y_center = (dy1 + dy2) / 2
            
            if normal_y_center is None:
                normal_y_center = current_y_center
            else:
                normal_y_center = 0.9 * normal_y_center + 0.1 * current_y_center
                
            if current_y_center - normal_y_center > 100: # Phát hiện chồm / cúi người
                cv2.putText(frame, "DANG CUI NGUOI LAYS DO?", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                is_distracted = True
                ear = 0.35
            else:
                cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
                cv2.putText(frame, "Driver", (dx1, dy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 3. Lọc mắt ưu tiên cho góc chéo (Chỉ cần 1 mắt nhắm)
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf < 0.4: continue
                    
                    ex1, ey1, ex2, ey2 = map(int, box.xyxy[0])
                    if cls_id == 3: mar = 0.8
                    
                    if cls_id in [0, 1]:
                        # Đảm bảo mắt nằm trong khung mặt của tài xế
                        if dx1 < ex1 and dy1 < ey1 and dx2 > ex2 and dy2 > ey2:
                            if cls_id == 0: eye_closed_detected = True
                            elif cls_id == 1: eye_open_detected = True

                # Nếu detect được dù chỉ 1 mắt nhắm, vẫn tính là buồn ngủ
                if eye_closed_detected: ear = 0.10
                elif eye_open_detected: ear = 0.35
                else: ear = 0.30

        # Nếu không bị khuất tầm nhìn, tiếp tục đưa dữ liệu vào Buffer
        if not is_distracted:
            pitch, yaw, roll = 0.0, 0.0, 0.0 
            geo_vector = [ear, mar, pitch, yaw, roll, 1.0] 
            spatial_vector = np.zeros(512) 
            
            geo_buffer.append(geo_vector)
            spatial_buffer.append(spatial_vector)
            
            if len(geo_buffer) > seq_length:
                geo_buffer.pop(0)
                spatial_buffer.pop(0)
        
        is_drowsy = False
        drowsy_prob = 0.0
        
        # 4. Kích hoạt Hybrid Model
        if len(geo_buffer) == seq_length and not is_distracted:
            with torch.no_grad():
                geo_tensor = torch.tensor([geo_buffer], dtype=torch.float32).to(device)
                spatial_tensor = torch.tensor([spatial_buffer], dtype=torch.float32).to(device)
                
                outputs = hybrid_brain(spatial_tensor, geo_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                drowsy_prob = probabilities[1].item()
                
                recent_ears = [frame_data[0] for frame_data in geo_buffer[-15:]]
                avg_ear = sum(recent_ears) / len(recent_ears)
                
                if avg_ear <= 0.15:
                    is_drowsy = True
                    drowsy_prob = 0.99 
                else:
                    is_drowsy = drowsy_prob > 0.7 
        
        # 5. Xử lý Logic Gợi ý Trạm dừng chân tự động
        if is_drowsy:
            awake_frames_count = 0 
            if not has_suggested_rest:
                print("--- Kích hoạt tìm trạm nghỉ tự động qua IP ---")
                lat, lon = get_location_by_ip()
                result = osm.find_rest_stop_auto_radius(lat, lon)
                socketio.emit('rest_stops_data', {
                    'success': True,
                    'stops': result['stops'],
                    'radius_used': result['radius_used'],
                    'auto_trigger': True
                })
                has_suggested_rest = True
        else:
            awake_frames_count += 1
            if awake_frames_count > 300: 
                has_suggested_rest = False
        
        # 6. Gửi dữ liệu lên Web Dashboard
        socketio.emit('update_data', {
            'ear': round(ear if not is_distracted else 0.35, 2),
            'pose_status': "Bình thường" if not is_distracted else "Mất tập trung",
            'is_drowsy': bool(is_drowsy),
            'occlusion': "Có" if is_distracted else "Không",
            'prob': f"{drowsy_prob*100:.1f}%"
        })
        
        # 7. Vẽ cảnh báo lên màn hình
        if is_drowsy:
            cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (0,0,255), 8)
            cv2.putText(frame, "CANH BAO NGU GAT!", (frame.shape[1]//2 - 150, frame.shape[0]//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        elif not is_distracted:
            cv2.putText(frame, f"EAR: {ear:.2f} | Ti le ngu gat: {drowsy_prob*100:.0f}%", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        with lock:
            output_frame = frame.copy()

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
    radius = int(data.get('radius', 60000))
    heading = data.get('heading')
    if heading is not None:
        heading = float(heading)

    result = osm.find_rest_stop_auto_radius(
        data['lat'],
        data['lon'],
        radius,
        heading
    )
    socketio.emit('rest_stops_data', {
        'success': True,
        'stops': result['stops'],
        'radius_used': result['radius_used']
    })

if __name__ == '__main__':
    t = Thread(target=ai_detection_thread)
    t.daemon = True
    t.start()
    print("--- Khởi động Web Server tại http://127.0.0.1:5000 ---")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
