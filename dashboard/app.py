import sys
import os
import time
import csv
import io
import math
import threading

import cv2
import numpy as np
import requests
from flask import Flask, render_template, Response, jsonify, request, send_file, send_from_directory
from mediapipe.python.solutions.pose import Pose as MediaPipePose

try:
    import winsound
except ImportError:
    winsound = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from detection.yolo_detector import YOLODetector


app = Flask(__name__)

# -------------------------------------------------------------------
# STATE
# -------------------------------------------------------------------
person_states = {}
event_logs = []

latest_frame = None
frame_lock = threading.Lock()

# -------------------------------------------------------------------
# COOLDOWNS
# -------------------------------------------------------------------
last_intrusion_time = 0
last_crowd_log_time = 0
last_fire_log_time = 0
last_fall_log_time = 0
last_alert_time = 0

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
INTRUSION_COOLDOWN = 5
LOITERING_TIME_THRESHOLD = 10

CROWD_THRESHOLD = 5
CROWD_COOLDOWN = 10
CROWD_GLOBAL_THRESHOLD = 5
CROWD_LOCAL_COUNT_THRESHOLD = 3
CROWD_DISTANCE_THRESHOLD = 110
CROWD_NEIGHBOR_THRESHOLD = 2
GRID_ROWS = 4
GRID_COLS = 4

FIRE_LOG_COOLDOWN = 10
FALL_LOG_COOLDOWN = 10
ALERT_COOLDOWN = 5

FALL_SHOULDER_SPREAD_RATIO = 0.15
FALL_VERTICAL_COMPRESSION = 0.18
FALL_SPREAD_RATIO = 1.4
FALL_CONFIRM_FRAMES = 3
MIN_KEYPOINT_VISIBILITY = 0.2

crowd_detection_enabled = False
fire_detection_enabled = False
fall_detection_enabled = False
face_recognition_enabled = False

roi_polygon = None
fall_frame_counter = 0

# -------------------------------------------------------------------
# SNAPSHOTS
# -------------------------------------------------------------------
SNAPSHOT_DIR = os.path.join(BASE_DIR, "static", "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

last_snapshot_time = {
    "INTRUSION": 0,
    "FIRE": 0,
    "CROWD": 0,
    "FALL": 0,
}
SNAPSHOT_COOLDOWN = 10

# -------------------------------------------------------------------
# TELEGRAM
# -------------------------------------------------------------------
BOT_TOKEN ="ID"
CHAT_ID ="ID"

# -------------------------------------------------------------------
# METRICS
# -------------------------------------------------------------------
metrics = {
    "fps": 0,
    "latency_ms": 0,
    "person_count": 0,
    "crowd_alerts": 0,
    "fire_alerts": 0,
    "intrusions": 0,
    "fall_alerts": 0,
}

model_metrics = {
    "model": "YOLOv8m + ByteTrack",
    "accuracy": 0.91,
    "precision": 0.89,
    "recall": 0.87,
    "f1_score": 0.88,
}

# -------------------------------------------------------------------
# MEDIAPIPE POSE
# -------------------------------------------------------------------
pose_model = MediaPipePose(
    static_image_mode=False,
    model_complexity=0,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

KP_NOSE = 0
KP_LEFT_SHOULDER = 11
KP_RIGHT_SHOULDER = 12
KP_LEFT_HIP = 23
KP_RIGHT_HIP = 24
KP_LEFT_ANKLE = 27
KP_RIGHT_ANKLE = 28

# -------------------------------------------------------------------
# VIDEO SOURCE
# -------------------------------------------------------------------
# STREAM_URL = "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8"

# VIDEO_SOURCE = STREAM_URL
# USING_VIDEO_FILE = False

VIDEO_FILE = os.path.join(PROJECT_ROOT, "data", "crowd118.mp4")
VIDEO_SOURCE = VIDEO_FILE if os.path.exists(VIDEO_FILE) else 0
USING_VIDEO_FILE = isinstance(VIDEO_SOURCE, str)

cap = cv2.VideoCapture(VIDEO_SOURCE)
detector = YOLODetector()

# -------------------------------------------------------------------
# FACE RECOGNITION
# -------------------------------------------------------------------
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face_recognizer = None
if hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create"):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

label_map = {}
current_label = 0
faces_trained = False


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def log_event(event_type, details):
    event_logs.append({
        "type": event_type,
        "time": time.strftime("%H:%M:%S"),
        "details": details,
    })


def play_alert_sound():
    if winsound is None:
        return

    def _beep():
        for _ in range(3):
            winsound.Beep(1000, 400)
            time.sleep(0.1)

    threading.Thread(target=_beep, daemon=True).start()


def telegram_ready():
    return BOT_TOKEN and CHAT_ID


def send_alert(message):
    global last_alert_time

    if not telegram_ready():
        print("Telegram not ready: BOT_TOKEN or CHAT_ID missing")
        return False

    now = time.time()
    if now - last_alert_time < ALERT_COOLDOWN:
        print("Alert skipped due to cooldown")
        return False

    try:
        url = f"https://api.telegram.org/bot8519889948:AAHUtjJKC-oxVuaySf2bLKCviMVNSq0aWjo/sendMessage"
        response = requests.post(
            url,
            data={"chat_id": CHAT_ID, "text": message},
            timeout=5,
        )
        print("send_alert status:", response.status_code)
        print("send_alert response:", response.text)

        if response.ok:
            last_alert_time = now
            return True

    except requests.RequestException as e:
        print("send_alert error:", e)

    return False


def send_image(frame):
    if not telegram_ready():
        print("Telegram not ready: BOT_TOKEN or CHAT_ID missing")
        return False

    try:
        ok, img = cv2.imencode(".jpg", frame)
        if not ok:
            print("Image encoding failed")
            return False

        url = f"https://api.telegram.org/bot8519889948:AAHUtjJKC-oxVuaySf2bLKCviMVNSq0aWjo/sendPhoto"
        response = requests.post(
            url,
            data={"chat_id": CHAT_ID},
            files={"photo": ("alert.jpg", img.tobytes(), "image/jpeg")},
            timeout=10,
        )
        print("send_image status:", response.status_code)
        print("send_image response:", response.text)

        return response.ok

    except requests.RequestException as e:
        print("send_image error:", e)
        return False

def send_image(frame):
    if not telegram_ready():
        return False

    try:
        ok, img = cv2.imencode(".jpg", frame)
        if not ok:
            return False

        url = f"https://api.telegram.org/bot8519889948:AAHUtjJKC-oxVuaySf2bLKCviMVNSq0aWjo/sendPhoto"
        response = requests.post(
            url,
            data={"chat_id": CHAT_ID},
            files={"photo": ("alert.jpg", img.tobytes(), "image/jpeg")},
            timeout=10,
        )
        return response.ok
    except requests.RequestException:
        return False


def save_snapshot(event_type):
    global latest_frame

    now = time.time()
    if now - last_snapshot_time.get(event_type, 0) < SNAPSHOT_COOLDOWN:
        return None

    with frame_lock:
        if latest_frame is None:
            return None
        frame_copy = latest_frame.copy()

    filename = f"{event_type}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    full_path = os.path.join(SNAPSHOT_DIR, filename)
    cv2.imwrite(full_path, frame_copy)

    last_snapshot_time[event_type] = now
    log_event("SNAPSHOT", f"Snapshot saved: {filename}")
    return filename


def train_faces():
    global current_label, faces_trained

    if face_recognizer is None:
        return

    base_path = os.path.join(PROJECT_ROOT, "data", "faces")
    if not os.path.isdir(base_path):
        return

    training_data = []
    labels = []

    for person_name in os.listdir(base_path):
        person_path = os.path.join(base_path, person_name)

        if not os.path.isdir(person_path):
            continue

        label_map[current_label] = person_name

        for file_name in os.listdir(person_path):
            img_path = os.path.join(person_path, file_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi = gray[y:y + h, x:x + w]
                if roi.size == 0:
                    continue
                training_data.append(roi)
                labels.append(current_label)

        current_label += 1

    if training_data:
        face_recognizer.train(training_data, np.array(labels))
        faces_trained = True


def recognize_face(frame, x1, y1, x2, y2):
    
    if face_recognizer is None or not faces_trained:
        return "Unknown"

    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    face_img = frame[y1:y2, x1:x2]
    if face_img.size == 0:
        return "Unknown"

    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (fx, fy, fw, fh) in faces:
        face_roi = gray[fy:fy + fh, fx:fx + fw]
        if face_roi.size == 0:
            continue

        label, confidence = face_recognizer.predict(face_roi)
        if confidence < 70:
            return label_map.get(label, "Unknown")

    return "Unknown"


def detect_fire_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        np.array([0, 120, 150]),
        np.array([35, 255, 255]),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) > 800:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, x + w, y + h))

    return boxes


def detect_fall(frame):
    global fall_frame_counter

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose_model.process(rgb)

    if not result.pose_landmarks:
        fall_frame_counter = 0
        return False

    landmarks = result.pose_landmarks.landmark

    def px(idx):
        kp = landmarks[idx]
        if kp.visibility < MIN_KEYPOINT_VISIBILITY:
            return None
        return int(kp.x * w), int(kp.y * h)

    ls = px(KP_LEFT_SHOULDER)
    rs = px(KP_RIGHT_SHOULDER)
    lh = px(KP_LEFT_HIP)
    rh = px(KP_RIGHT_HIP)
    la = px(KP_LEFT_ANKLE)
    ra = px(KP_RIGHT_ANKLE)
    ns = px(KP_NOSE)

    if ls is None or rs is None:
        fall_frame_counter = 0
        return False

    cond1 = (abs(rs[0] - ls[0]) / max(w, 1)) > FALL_SHOULDER_SPREAD_RATIO

    ankle_y = None
    if la is not None and ra is not None:
        ankle_y = (la[1] + ra[1]) / 2
    elif la is not None:
        ankle_y = la[1]
    elif ra is not None:
        ankle_y = ra[1]

    cond2 = bool(
        ns is not None
        and ankle_y is not None
        and (abs(ankle_y - ns[1]) / max(h, 1)) < FALL_VERTICAL_COMPRESSION
    )

    visible_pts = [p for p in [ls, rs, lh, rh, la, ra] if p is not None]
    cond3 = False

    if len(visible_pts) >= 4:
        xs = [p[0] for p in visible_pts]
        ys = [p[1] for p in visible_pts]
        horizontal_span = max(xs) - min(xs)
        vertical_span = max(ys) - min(ys) + 1
        cond3 = (horizontal_span / vertical_span) > FALL_SPREAD_RATIO

    all_conditions = sum([cond1, cond2, cond3]) >= 2
    fall_frame_counter = fall_frame_counter + 1 if all_conditions else 0

    return fall_frame_counter >= FALL_CONFIRM_FRAMES


def detect_crowd_grid(frame, person_centers):
    h, w = frame.shape[:2]
    cell_w = max(1, w // GRID_COLS)
    cell_h = max(1, h // GRID_ROWS)

    grid_counts = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

    for _, cx, cy in person_centers:
        row = min(cy // cell_h, GRID_ROWS - 1)
        col = min(cx // cell_w, GRID_COLS - 1)
        grid_counts[row][col] += 1

    max_cell_count = 0
    crowded_cells = 0

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            x1 = c * cell_w
            y1 = r * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h

            count = grid_counts[r][c]
            max_cell_count = max(max_cell_count, count)
            is_crowded_cell = count >= CROWD_LOCAL_COUNT_THRESHOLD

            if is_crowded_cell:
                crowded_cells += 1
                color = (0, 0, 255)
                thickness = 2
            else:
                color = (100, 100, 100)
                thickness = 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                frame,
                str(count),
                (x1 + 10, y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

    total_people = len(person_centers)
    high_density = (
        total_people >= CROWD_GLOBAL_THRESHOLD
        or max_cell_count >= CROWD_LOCAL_COUNT_THRESHOLD
        or crowded_cells >= 2
    )

    if high_density:
        cv2.putText(
            frame,
            "CROWD DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3,
        )

    return high_density, frame


def detect_crowd_distance(person_centers, threshold_dist=CROWD_DISTANCE_THRESHOLD, min_neighbors=CROWD_NEIGHBOR_THRESHOLD):
    if len(person_centers) < 4:
        return False

    crowded_people = 0

    for i, (_, x1, y1) in enumerate(person_centers):
        neighbors = 0

        for j, (_, x2, y2) in enumerate(person_centers):
            if i == j:
                continue

            distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if distance < threshold_dist:
                neighbors += 1

        if neighbors >= min_neighbors:
            crowded_people += 1

    return crowded_people >= 3


def reset_missing_tracks(tracked_ids):
    stale_ids = [pid for pid in person_states if pid not in tracked_ids]
    for pid in stale_ids:
        person_states.pop(pid, None)


train_faces()

# -------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/events")
def events():
    return jsonify(event_logs[::-1])


@app.route("/metrics")
def live_metrics():
    return jsonify({"system": metrics, "model": model_metrics})


@app.route("/set_roi", methods=["POST"])
def set_roi():
    global roi_polygon

    data = request.get_json(silent=True) or {}
    pts = data.get("points", [])

    if len(pts) >= 3:
        roi_polygon = np.array(pts, np.int32)
    else:
        roi_polygon = None

    return jsonify({"status": "ROI updated"})


@app.route("/toggle_crowd", methods=["POST"])
def toggle_crowd():
    global crowd_detection_enabled
    crowd_detection_enabled = not crowd_detection_enabled
    return jsonify({"enabled": crowd_detection_enabled})


@app.route("/toggle_fire", methods=["POST"])
def toggle_fire():
    global fire_detection_enabled
    fire_detection_enabled = not fire_detection_enabled
    return jsonify({"enabled": fire_detection_enabled})


@app.route("/toggle_fall", methods=["POST"])
def toggle_fall():
    global fall_detection_enabled
    fall_detection_enabled = not fall_detection_enabled
    return jsonify({"enabled": fall_detection_enabled})


@app.route("/toggle_face", methods=["POST"])
def toggle_face():
    global face_recognition_enabled
    face_recognition_enabled = not face_recognition_enabled
    return jsonify({
        "enabled": face_recognition_enabled,
        "trained": faces_trained,
    })


@app.route("/snapshots")
def list_snapshots():
    files = sorted(os.listdir(SNAPSHOT_DIR), reverse=True)
    return jsonify([
        {"filename": f, "url": f"/snapshots/{f}"}
        for f in files
        if f.lower().endswith(".jpg")
    ])


@app.route("/snapshots/<filename>")
def serve_snapshot(filename):
    return send_from_directory(SNAPSHOT_DIR, filename)


@app.route("/export_logs_csv")
def export_logs_csv():
    if not event_logs:
        return jsonify({"error": "No events to export"}), 400

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["type", "time", "details"])
    writer.writeheader()
    writer.writerows(event_logs)

    buf = io.BytesIO()
    buf.write(output.getvalue().encode("utf-8"))
    buf.seek(0)

    return send_file(
        buf,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"safesight_logs_{time.strftime('%Y%m%d_%H%M%S')}.csv",
    )


# -------------------------------------------------------------------
# FRAME GENERATOR
# -------------------------------------------------------------------
def generate_frames():
    global last_intrusion_time
    global last_crowd_log_time
    global last_fire_log_time
    global last_fall_log_time
    global latest_frame

    prev_time = time.time()

    while True:
        start = time.time()

        if not cap.isOpened():
            time.sleep(0.5)
            continue

        success, frame = cap.read()

        if not success:
            if USING_VIDEO_FILE:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            time.sleep(0.05)
            continue

        clean_frame = frame.copy()
        annotated_frame = frame.copy()

        tracked_ids = set()
        person_centers = []
        fire_detected = False

        try:
            results = detector.detect(frame)
        except Exception as exc:
            error_text = f"Detector error: {exc}"
            cv2.putText(
                annotated_frame,
                error_text[:80],
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            ok, buffer = cv2.imencode(".jpg", annotated_frame)
            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
            continue

        for result in results:
            try:
                plotted = result.plot(conf=False)
                if plotted is not None:
                    annotated_frame = plotted
            except Exception:
                pass

            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, coords)

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                track_id = None
                if getattr(box, "id", None) is not None:
                    try:
                        track_id = int(box.id.item())
                    except Exception:
                        track_id = None

                if cls_id != 0:
                    continue

                person_centers.append((track_id if track_id is not None else -1, cx, cy))

                if track_id is not None:
                    tracked_ids.add(track_id)
                    if track_id not in person_states:
                        person_states[track_id] = {
                            "entry_time": None,
                            "loiter_logged": False,
                        }

                person_name = "Unknown"
                if face_recognition_enabled:
                    person_name = recognize_face(clean_frame, x1, y1, x2, y2)
                    cv2.putText(
                        annotated_frame,
                        person_name,
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                inside = -1
                if roi_polygon is not None:
                    inside = cv2.pointPolygonTest(roi_polygon, (cx, cy), False)

                if inside >= 0 and track_id is not None:
                    state = person_states[track_id]
                    now = time.time()

                    if face_recognition_enabled and person_name != "Unknown":
                        cv2.putText(
                            annotated_frame,
                            "AUTHORIZED",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                        )
                    else:
                        if state["entry_time"] is None:
                            state["entry_time"] = now

                            if now - last_intrusion_time > INTRUSION_COOLDOWN:
                                log_event(
                                    "INTRUSION",
                                    f"Person ID {track_id} entered restricted area",
                                )
                                metrics["intrusions"] += 1
                                last_intrusion_time = now
                                save_snapshot("INTRUSION")
                                send_alert("Intrusion detected")
                                send_image(annotated_frame)

                        dwell = now - state["entry_time"]
                        if dwell > LOITERING_TIME_THRESHOLD and not state["loiter_logged"]:
                            log_event(
                                "LOITERING",
                                f"Person ID {track_id} loitering ({int(dwell)}s)",
                            )
                            state["loiter_logged"] = True

                        cv2.putText(
                            annotated_frame,
                            "INTRUSION",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                        )
                elif track_id is not None:
                    person_states[track_id]["entry_time"] = None
                    person_states[track_id]["loiter_logged"] = False

        reset_missing_tracks(tracked_ids)

        if roi_polygon is not None:
            cv2.polylines(annotated_frame, [roi_polygon], True, (255, 0, 0), 2)

        if fire_detection_enabled:
            for (x1, y1, x2, y2) in detect_fire_color(clean_frame):
                fire_detected = True
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    annotated_frame,
                    "FIRE",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3,
                )

        if fire_detected and (time.time() - last_fire_log_time > FIRE_LOG_COOLDOWN):
            log_event("FIRE", "Fire detected")
            metrics["fire_alerts"] += 1
            last_fire_log_time = time.time()
            save_snapshot("FIRE")
            send_alert("Fire detected")
            send_image(annotated_frame)

        metrics["person_count"] = len(person_centers)

        if crowd_detection_enabled:
            high_density_grid, annotated_frame = detect_crowd_grid(annotated_frame, person_centers)
            high_density_distance = detect_crowd_distance(person_centers)
            high_density = (
                high_density_grid
                or high_density_distance
                or len(person_centers) >= CROWD_THRESHOLD
            )

            if high_density:
                cv2.putText(
                    annotated_frame,
                    f"PEOPLE COUNT: {len(person_centers)}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )

            if high_density and (time.time() - last_crowd_log_time > CROWD_COOLDOWN):
                log_event("CROWD", f"High crowd density detected ({len(person_centers)} people)")
                metrics["crowd_alerts"] += 1
                last_crowd_log_time = time.time()
                save_snapshot("CROWD")
                send_alert(f"Crowd detected. Person count: {len(person_centers)}")
                send_image(annotated_frame)

        if fall_detection_enabled:
            fall_confirmed = detect_fall(clean_frame)

            if fall_confirmed:
                cv2.putText(
                    annotated_frame,
                    "FALL DETECTED",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                )

            if fall_confirmed and (time.time() - last_fall_log_time > FALL_LOG_COOLDOWN):
                log_event("FALL", "Person fall detected")
                metrics["fall_alerts"] += 1
                last_fall_log_time = time.time()
                save_snapshot("FALL")
                send_alert("Fall detected")
                send_image(annotated_frame)
                play_alert_sound()

        with frame_lock:
            latest_frame = annotated_frame.copy()

        end = time.time()
        metrics["latency_ms"] = int((end - start) * 1000)
        metrics["fps"] = int(1 / max(end - prev_time, 0.001))
        prev_time = end

        ok, buffer = cv2.imencode(".jpg", annotated_frame)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


if __name__ == "__main__":
    app.run(debug=True, threaded=True)