import cv2
import numpy as np
import time
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from threading import Thread, Lock
import mediapipe as mp
import winsound

from services.osm_service import OSMService

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

lock = Lock()
output_frame = None

osm = OSMService()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# ===== TĂNG SÁNG TỰ ĐỘNG =====
def enhance_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) < 80:
        frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=40)
    return frame

# ===== EAR =====
def calculate_EAR(landmarks, w, h):
    idx = [33,160,158,133,153,144]
    pts = [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in idx]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A+B)/(2.0*C)

# ===== MAR =====
def calculate_MAR(landmarks, w, h):
    pts = [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in [13,14,78,308]]
    return np.linalg.norm(np.array(pts[0])-np.array(pts[1])) / np.linalg.norm(np.array(pts[2])-np.array(pts[3]))

# ===== HEAD POSE =====
def head_pose(landmarks, h):
    return int(landmarks[152].y*h) - int(landmarks[1].y*h)

# ===== AI THREAD =====
def ai_thread():
    global output_frame
    cap = cv2.VideoCapture(0)

    ear_buffer = []
    base_ear = None
    calibrate = 30

    eye_closed_start = None
    yawn_start = None
    drowsy_score = 0

    last_alert = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = enhance_frame(frame)

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        ear, mar, pose = 0.3, 0.1, 0
        eye_visible = True

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            ear = calculate_EAR(lm, w, h)
            mar = calculate_MAR(lm, w, h)
            pose = head_pose(lm, h)

            if ear < 0.05:
                eye_visible = False

        # ===== SMOOTH EAR =====
        ear_buffer.append(ear)
        if len(ear_buffer) > 10:
            ear_buffer.pop(0)

        avg_ear = np.mean(ear_buffer)

        # ===== CALIBRATE =====
        if base_ear is None and len(ear_buffer) >= calibrate:
            base_ear = np.mean(ear_buffer)

        threshold = base_ear*0.75 if base_ear else 0.22

        # ===== NHẮM MẮT =====
        if avg_ear < threshold:
            if eye_closed_start is None:
                eye_closed_start = time.time()
        else:
            eye_closed_start = None

        eye_closed_duration = 0
        if eye_closed_start:
            eye_closed_duration = time.time() - eye_closed_start

        # ===== NGÁP =====
        if mar > 0.7:
            if yawn_start is None:
                yawn_start = time.time()
        else:
            yawn_start = None

        yawn_duration = 0
        if yawn_start:
            yawn_duration = time.time() - yawn_start

        # ===== AI SCORE =====
        if eye_closed_duration > 2:
            drowsy_score += 2

        if yawn_duration > 1.5:
            drowsy_score += 1.5

        if pose > 120:
            drowsy_score += 2

        # giảm nếu tỉnh
        if eye_closed_duration < 1 and yawn_duration < 1 and pose < 100:
            drowsy_score -= 1

        drowsy_score = max(0, min(drowsy_score, 10))

        is_drowsy = drowsy_score >= 5

        # ===== CẢNH BÁO =====
        now = time.time()
        if is_drowsy and now - last_alert > 2:
            winsound.Beep(1000, 300)
            last_alert = now

        # ===== UI =====
        if is_drowsy:
            cv2.rectangle(frame,(0,0),(w,h),(0,0,255),10)
            cv2.putText(frame,"NGU GAT!",(50,100),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

        cv2.putText(frame,f"EAR:{avg_ear:.2f}",(10,30),0,0.7,(0,255,0),2)
        cv2.putText(frame,f"Score:{drowsy_score:.1f}",(10,60),0,0.7,(0,255,255),2)
        cv2.putText(frame,f"Eye:{eye_closed_duration:.1f}s",(10,90),0,0.7,(255,255,0),2)
        cv2.putText(frame,f"Yawn:{yawn_duration:.1f}s",(10,120),0,0.7,(255,0,255),2)

        with lock:
            output_frame = frame.copy()

        socketio.emit('update_data',{
            'ear': avg_ear,
            'head_pose': pose,
            'eye_visible': eye_visible,
            'is_drowsy': is_drowsy
        })

# ===== ROUTES =====
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video():
    def gen():
        while True:
            if output_frame is None:
                continue
            with lock:
                _, buf = cv2.imencode('.jpg', output_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                   bytearray(buf) + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ===== MAP =====
@socketio.on('find_stops_request')
def find_stops(data):
    stops = osm.find_nearest_rest_stop(data['lat'], data['lon'])
    socketio.emit('rest_stops_data', {'stops': stops})

# ===== MAIN =====
if __name__ == '__main__':
    Thread(target=ai_thread, daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5000)