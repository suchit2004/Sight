import sys
import os
import time
import cv2
import numpy as np
import csv
import io
import threading
import winsound
import mediapipe
from mediapipe.python.solutions.pose import Pose as MediaPipePose
from flask import Flask, render_template, Response, jsonify, request, send_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from detection.yolo_detector import YOLODetector

app = Flask(__name__)

# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────
person_states = {}
event_logs = []

# ─────────────────────────────────────────────
# COOLDOWNS
# ─────────────────────────────────────────────
last_intrusion_time = 0
last_crowd_log_time = 0
last_fire_log_time = 0
last_fall_log_time = 0

# ─────────────────────────────────────────────
# INTRUSION
# ─────────────────────────────────────────────
INTRUSION_COOLDOWN = 5
LOITERING_TIME_THRESHOLD = 10

# ─────────────────────────────────────────────
# CROWD
# ─────────────────────────────────────────────
CROWD_THRESHOLD = 5
CROWD_COOLDOWN = 10
crowd_detection_enabled = False

# ─────────────────────────────────────────────
# FIRE
# ─────────────────────────────────────────────
fire_detection_enabled = False
FIRE_LOG_COOLDOWN = 10

# ─────────────────────────────────────────────
# FALL
# ─────────────────────────────────────────────
fall_detection_enabled = False
FALL_LOG_COOLDOWN = 10
FALL_SHOULDER_SPREAD_RATIO = 0.15
FALL_VERTICAL_COMPRESSION = 0.18
FALL_SPREAD_RATIO = 1.4
FALL_CONFIRM_FRAMES = 3
MIN_KEYPOINT_VISIBILITY = 0.2

# ─────────────────────────────────────────────
# ROI
# ─────────────────────────────────────────────
roi_polygon = None

# ─────────────────────────────────────────────
# SNAPSHOTS
# ─────────────────────────────────────────────
SNAPSHOT_DIR = os.path.join(BASE_DIR, "static", "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

latest_frame = None
frame_lock = threading.Lock()

last_snapshot_time = {
    "INTRUSION": 0,
    "FIRE": 0,
    "CROWD": 0,
    "FALL": 0,
}

SNAPSHOT_COOLDOWN = 10
fall_frame_counter = 0

# ─────────────────────────────────────────────
# MEDIAPIPE
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# VIDEO
# ─────────────────────────────────────────────
cap = cv2.VideoCapture("../data/CLIP1.mp4")
detector = YOLODetector()

# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def play_alert_sound():
    def _beep():
        for _ in range(3):
            winsound.Beep(1000, 400)
            time.sleep(0.1)
    threading.Thread(target=_beep, daemon=True).start()


def save_snapshot(event_type):
    global latest_frame

    now = time.time()

    if now - last_snapshot_time.get(event_type, 0) < SNAPSHOT_COOLDOWN:
        return

    with frame_lock:
        if latest_frame is None:
            return
        frame_copy = latest_frame.copy()

    filename = f"{event_type}_{time.strftime('%H%M%S')}.jpg"
    cv2.imwrite(os.path.join(SNAPSHOT_DIR, filename), frame_copy)

    last_snapshot_time[event_type] = now

    event_logs.append({
        "type": "SNAPSHOT",
        "time": time.strftime("%H:%M:%S"),
        "details": f"Snapshot saved: {filename}",
    })


# ─────────────────────────────────────────────
# FIRE DETECTION
# ─────────────────────────────────────────────

def detect_fire_color(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        np.array([0, 120, 150]),
        np.array([35, 255, 255])
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8)
    )

    cnts, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for c in cnts:

        if cv2.contourArea(c) > 800:

            x, y, w, h = cv2.boundingRect(c)

            boxes.append((x, y, x + w, y + h))

    return boxes


# ─────────────────────────────────────────────
# FALL DETECTION
# ─────────────────────────────────────────────

def detect_fall(frame):

    global fall_frame_counter

    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = pose_model.process(rgb)

    if not result.pose_landmarks:

        fall_frame_counter = 0
        return False, frame

    lm = result.pose_landmarks.landmark

    def px(idx):

        kp = lm[idx]

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
        return False, frame

    cond1 = (abs(rs[0] - ls[0]) / w) > FALL_SHOULDER_SPREAD_RATIO

    ankle_y = (
        (la[1] + ra[1]) / 2 if la and ra
        else la[1] if la
        else ra[1] if ra
        else None
    )

    cond2 = bool(
        ns and ankle_y and
        (abs(ankle_y - ns[1]) / h) < FALL_VERTICAL_COMPRESSION
    )

    visible_pts = [
        p for p in [ls, rs, lh, rh, la, ra] if p is not None
    ]

    cond3 = False

    if len(visible_pts) >= 4:

        xs = [p[0] for p in visible_pts]
        ys = [p[1] for p in visible_pts]

        cond3 = (
            (max(xs) - min(xs))
            /
            (max(ys) - min(ys) + 1)
        ) > FALL_SPREAD_RATIO

    conditions = [cond1, cond2, cond3]
    all_conditions = sum(conditions) >= 2

    fall_frame_counter = (
        fall_frame_counter + 1 if all_conditions else 0
    )

    fall_confirmed = fall_frame_counter >= FALL_CONFIRM_FRAMES

    if fall_confirmed:

        cv2.putText(
            frame,
            "FALL DETECTED",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0, 0, 255),
            3
        )

    return fall_confirmed, frame


# ─────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────

@app.route("/set_roi", methods=["POST"])
def set_roi():

    global roi_polygon

    pts = request.json.get("points", [])

    roi_polygon = (
        np.array(pts, np.int32)
        if len(pts) >= 3
        else None
    )

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


@app.route("/export_logs_csv")
def export_logs_csv():

    if not event_logs:
        return jsonify({"error": "No events to export"}), 400

    output = io.StringIO()

    writer = csv.DictWriter(
        output,
        fieldnames=["type", "time", "details"]
    )

    writer.writeheader()

    writer.writerows(event_logs)

    buf = io.BytesIO()
    buf.write(output.getvalue().encode("utf-8"))
    buf.seek(0)

    return send_file(
        buf,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"safesight_logs_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/events")
def events():
    return jsonify(event_logs[::-1])


@app.route("/metrics")
def live_metrics():
    return jsonify({"system": metrics, "model": model_metrics})


# ─────────────────────────────────────────────
# FRAME GENERATOR
# ─────────────────────────────────────────────

def generate_frames():

    global last_intrusion_time
    global last_crowd_log_time
    global last_fire_log_time
    global last_fall_log_time
    global latest_frame

    prev_time = time.time()

    while True:

        start = time.time()

        success, frame = cap.read()

        if not success:

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        results = detector.detect(frame)

        clean_frame = frame.copy()

        tracked_ids = set()

        fire_detected = False

        for result in results:

            frame = result.plot(conf=False)

            if roi_polygon is not None:

                cv2.polylines(
                    frame,
                    [roi_polygon],
                    True,
                    (255, 0, 0),
                    2
                )

            for box in result.boxes:

                if box.id is None:
                    continue

                cls_id = int(box.cls[0])

                track_id = int(box.id.item())

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                if cls_id == 0:

                    tracked_ids.add(track_id)

                    if track_id not in person_states:

                        person_states[track_id] = {
                            "entry_time": None,
                            "loiter_logged": False
                        }

                    state = person_states[track_id]

                    inside = -1

                    if roi_polygon is not None:

                        inside = cv2.pointPolygonTest(
                            roi_polygon,
                            (cx, cy),
                            False
                        )

                    if inside >= 0:

                        now = time.time()

                        if state["entry_time"] is None:

                            state["entry_time"] = now

                            if now - last_intrusion_time > INTRUSION_COOLDOWN:

                                event_logs.append({
                                    "type": "INTRUSION",
                                    "time": time.strftime("%H:%M:%S"),
                                    "details": f"Person ID {track_id} entered restricted area",
                                })

                                metrics["intrusions"] += 1

                                last_intrusion_time = now

                                save_snapshot("INTRUSION")

                        dwell = now - state["entry_time"]

                        if dwell > LOITERING_TIME_THRESHOLD and not state["loiter_logged"]:

                            event_logs.append({
                                "type": "LOITERING",
                                "time": time.strftime("%H:%M:%S"),
                                "details": f"Person ID {track_id} loitering ({int(dwell)}s)",
                            })

                            state["loiter_logged"] = True

                        cv2.putText(
                            frame,
                            "INTRUSION",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2
                        )

                    else:

                        state["entry_time"] = None
                        state["loiter_logged"] = False

        if fire_detection_enabled:

            for (x1, y1, x2, y2) in detect_fire_color(frame):

                fire_detected = True

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                cv2.putText(
                    frame,
                    "FIRE",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

        if fire_detected and time.time() - last_fire_log_time > FIRE_LOG_COOLDOWN:

            event_logs.append({
                "type": "FIRE",
                "time": time.strftime("%H:%M:%S"),
                "details": "Fire detected"
            })

            metrics["fire_alerts"] += 1

            last_fire_log_time = time.time()

            save_snapshot("FIRE")

        metrics["person_count"] = len(tracked_ids)

        if crowd_detection_enabled and metrics["person_count"] >= CROWD_THRESHOLD:

            if time.time() - last_crowd_log_time > CROWD_COOLDOWN:

                event_logs.append({
                    "type": "CROWD",
                    "time": time.strftime("%H:%M:%S"),
                    "details": f"High crowd density ({metrics['person_count']})"
                })

                metrics["crowd_alerts"] += 1

                last_crowd_log_time = time.time()

                save_snapshot("CROWD")

        if fall_detection_enabled:

            fall_confirmed, pose_frame = detect_fall(clean_frame)

# overlay fall text on main frame if detected
            if fall_confirmed:
             cv2.putText(
             frame,
             "FALL DETECTED",
             (30, 60),
             cv2.FONT_HERSHEY_SIMPLEX,
             1.4,
             (0, 0, 255),
             3
             )

            if fall_confirmed and time.time() - last_fall_log_time > FALL_LOG_COOLDOWN:

                event_logs.append({
                    "type": "FALL",
                    "time": time.strftime("%H:%M:%S"),
                    "details": "Person fall detected"
                })

                metrics["fall_alerts"] += 1

                last_fall_log_time = time.time()

                save_snapshot("FALL")

                play_alert_sound()

        with frame_lock:
            latest_frame = frame.copy()

        end = time.time()

        metrics["latency_ms"] = int((end - start) * 1000)

        metrics["fps"] = int(1 / max(end - prev_time, 0.001))

        prev_time = end

        _, buffer = cv2.imencode(".jpg", frame)

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


if __name__ == "__main__":
    app.run(debug=True)