# app.py

import cv2
import platform
import time
from threading import Lock, Thread

import mediapipe as mp
from flask import Flask, Response, jsonify, render_template
from flask_socketio import SocketIO

from config import AppConfig, MapConfig
from services.alert_logger import AlertLogger
from services.driver_selector import DriverSelector
from services.drowsiness_detector import DrowsinessDetector
from services.location_service import LocationService
from services.osm_service import OSMService


app = Flask(__name__)
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

frame_lock = Lock()
output_frame = None
latest_detection_result = None

osm_service = OSMService()
location_service = LocationService()
driver_selector = DriverSelector()
drowsiness_detector = DrowsinessDetector()
alert_logger = AlertLogger()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=3,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def beep_alert():
    try:
        if platform.system().lower() == "windows":
            import winsound
            winsound.Beep(1200, 350)
        else:
            print("\a")
    except Exception:
        print("ALERT!")


def enhance_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()

    if brightness < 75:
        frame = cv2.convertScaleAbs(frame, alpha=1.55, beta=45)

    return frame


def find_stops_auto_radius(lat, lon, max_radius=MapConfig.DEFAULT_RADIUS, heading=None):
    return osm_service.find_rest_stop_auto_radius(
        lat,
        lon,
        max_radius,
        heading
    )


def draw_driver_roi(frame):
    h, w, _ = frame.shape
    config = driver_selector.get_config()
    active_roi = driver_selector.get_active_roi()
    rx1, ry1, rx2, ry2 = driver_selector.denormalize_roi(active_roi, w, h)

    color = (255, 255, 0) if config["lock_driver"] else (120, 120, 120)
    label = "LOCKED DRIVER ROI" if config["has_custom_roi"] else "DEFAULT DRIVER ROI"

    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 2)
    cv2.putText(
        frame,
        label,
        (rx1, max(25, ry1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2
    )


def draw_state(frame, result, face_count, selected_face_box):
    h, w, _ = frame.shape
    state = result["state"]

    if state == "DROWSY_CONFIRMED":
        color = (0, 0, 255)
        label = "NGU GAT!"
        cv2.rectangle(frame, (0, 0), (w, h), color, 10)
    elif state == "WARNING_LEVEL_1":
        color = (0, 165, 255)
        label = "CANH BAO MET MOI"
    elif state == "WARNING_SUNGLASSES_MODE":
        color = (0, 165, 255)
        label = "CANH BAO - KINH RAM"
    elif state == "WARNING_MASK_MODE":
        color = (0, 165, 255)
        label = "CANH BAO - KHAU TRANG"
    elif state == "SUNGLASSES_MODE":
        color = (255, 255, 0)
        label = "CHE DO KINH RAM"
    elif state == "MASK_MODE":
        color = (255, 255, 0)
        label = "CHE DO KHAU TRANG"
    elif state == "CAMERA_BAD":
        color = (255, 0, 0)
        label = "GOC CAMERA KEM"
    elif state == "NO_DRIVER_IN_ROI":
        color = (180, 180, 180)
        label = "CHUA THAY TAI XE TRONG VUNG"
    elif state == "NO_FACE":
        color = (180, 180, 180)
        label = "KHONG THAY MAT"
    else:
        color = (0, 255, 0)
        label = "TINH TAO"

    cv2.putText(
        frame,
        label,
        (30, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        3
    )

    if selected_face_box is not None:
        x1, y1, x2, y2 = selected_face_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.putText(
            frame,
            "DRIVER",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 180, 255),
            2
        )

    overlays = [
        (f"EAR: {result['ear']:.2f}", h - 180, (0, 255, 0)),
        (f"MAR: {result['mar']:.2f}", h - 150, (255, 0, 255)),
        (f"Pose: {result['head_pose']:.2f}", h - 120, (255, 255, 0)),
        (f"Score: {result['drowsy_score']:.1f}", h - 90, (0, 255, 255)),
        (f"Mode: {result['detection_mode']}", h - 60, (255, 255, 255)),
        (f"Faces: {face_count} Conf: {result['confidence']:.2f}", h - 30, (255, 255, 255))
    ]

    for text, y, color in overlays:
        cv2.putText(
            frame,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2
        )


def camera_loop():
    global output_frame
    global latest_detection_result

    cap = cv2.VideoCapture(AppConfig.CAMERA_INDEX)

    while True:
        ret, frame = cap.read()

        if not ret:
            time.sleep(0.05)
            continue

        frame = enhance_frame(frame)
        h, w, _ = frame.shape
        draw_driver_roi(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb)

        face_count = 0
        selected_face_box = None
        selected_landmarks = None

        if face_results.multi_face_landmarks:
            face_count = len(face_results.multi_face_landmarks)
            selected = driver_selector.select_driver_face(
                face_results.multi_face_landmarks,
                w,
                h
            )

            if selected:
                selected_face_box = selected["box"]
                selected_landmarks = selected["face"].landmark

        result = drowsiness_detector.update(
            face_count=face_count,
            selected_face_box=selected_face_box,
            landmarks=selected_landmarks,
            frame_w=w,
            frame_h=h,
            frame=frame
        )

        latest_detection_result = result.copy()
        draw_state(frame, result, face_count, selected_face_box)

        if result["is_drowsy"] and result["can_alert"]:
            Thread(target=beep_alert, daemon=True).start()
            alert_logger.add_alert(result)
            drowsiness_detector.mark_alerted()

        with frame_lock:
            output_frame = frame.copy()

        driver_config = driver_selector.get_config()
        socketio.emit("update_data", {
            **result,
            "face_count": face_count,
            "driver_side": driver_config["side"],
            "driver_lock": driver_config["lock_driver"],
            "driver_custom_roi": driver_config["has_custom_roi"],
            "driver_selected": selected_face_box is not None
        })

        time.sleep(0.03)


@app.route("/")
def index():
    return render_template(
        "index.html",
        alert_logs=alert_logger.get_recent_logs(AppConfig.ALERT_HISTORY_LIMIT),
        alert_stats=alert_logger.get_stats()
    )


@app.route("/video_feed")
def video_feed():
    def gen():
        global output_frame

        while True:
            if output_frame is None:
                time.sleep(0.05)
                continue

            with frame_lock:
                success, buffer = cv2.imencode(".jpg", output_frame)

            if not success:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    return Response(
        gen(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/location")
def get_location_by_ip():
    location = location_service.get_location_by_ip()

    if location is None:
        return jsonify({
            "success": False,
            "message": "Không lấy được vị trí bằng IP"
        }), 500

    return jsonify({
        "success": True,
        "location": location
    })


@app.route("/api/alerts")
def get_alert_logs():
    return jsonify({
        "success": True,
        "stats": alert_logger.get_stats(),
        "logs": alert_logger.get_recent_logs(AppConfig.ALERT_HISTORY_LIMIT)
    })


@socketio.on("find_stops_request")
def find_stops(data):
    try:
        lat = float(data.get("lat"))
        lon = float(data.get("lon"))
        radius = int(data.get("radius", MapConfig.DEFAULT_RADIUS))
        heading = data.get("heading")

        if heading is not None:
            heading = float(heading)

        result = find_stops_auto_radius(lat, lon, radius, heading)
        radius_used = result["radius_used"]
        stops = result["stops"]

        location = {
            "lat": lat,
            "lon": lon,
            "radius_request": radius,
            "radius_used": radius_used,
            "heading": heading
        }

        alert_logger.update_latest_location(location=location, stops=stops)
        socketio.emit("rest_stops_data", {
            "success": True,
            "lat": lat,
            "lon": lon,
            "radius_used": radius_used,
            "heading": heading,
            "stops": stops
        })

    except Exception as e:
        socketio.emit("rest_stops_data", {
            "success": False,
            "message": str(e),
            "stops": []
        })


@socketio.on("update_driver_config")
def update_driver_config(data):
    config = driver_selector.update_config(
        side=data.get("side"),
        lock_driver=data.get("lock_driver")
    )

    socketio.emit("driver_config_updated", {
        "success": True,
        "message": "Đã cập nhật vùng ghế lái.",
        "side": config["side"],
        "lock_driver": config["lock_driver"],
        "custom_roi": config["has_custom_roi"]
    })


@socketio.on("lock_current_driver")
def lock_current_driver():
    result = driver_selector.lock_current_driver()
    socketio.emit("driver_config_updated", {
        "success": result["success"],
        "message": result["message"],
        "side": result["side"],
        "lock_driver": result["lock_driver"],
        "custom_roi": result["has_custom_roi"]
    })


@socketio.on("unlock_driver_roi")
def unlock_driver_roi():
    result = driver_selector.unlock_driver_roi()
    socketio.emit("driver_config_updated", {
        "success": result["success"],
        "message": result["message"],
        "side": result["side"],
        "lock_driver": result["lock_driver"],
        "custom_roi": result["has_custom_roi"]
    })


if __name__ == "__main__":
    Thread(target=camera_loop, daemon=True).start()
    socketio.run(
        app,
        host=AppConfig.HOST,
        port=AppConfig.PORT,
        debug=AppConfig.DEBUG
    )
