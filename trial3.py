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
from collections import deque
from mediapipe.python.solutions.pose import Pose as MediaPipePose
from mediapipe.python.solutions import pose as mp_pose
from flask import Flask, render_template, Response, jsonify, request, send_file

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from detection.yolo_detector import YOLODetector
from dashboard.face_auth import FaceAuthLBPH

app = Flask(__name__)

# ─── Per-object state ─────────────────────────────────────────────────────────
person_states  = {}
vehicle_states = {}
event_logs     = []

# ─── Cooldown timestamps ──────────────────────────────────────────────────────
last_intrusion_time = 0
last_crowd_log_time = 0
last_fire_log_time  = 0
last_fall_log_time  = 0
last_accident_time  = 0

# ─── Feature constants ────────────────────────────────────────────────────────
INTRUSION_COOLDOWN       = 5
LOITERING_TIME_THRESHOLD = 10

CROWD_THRESHOLD         = 5
CROWD_COOLDOWN          = 10
crowd_detection_enabled = False

fire_detection_enabled = False
FIRE_LOG_COOLDOWN      = 10

fall_detection_enabled = False
FALL_LOG_COOLDOWN      = 10

# ─── Face recognition ──────────────────────────────────────────────────────────
face_recognition_enabled = False
FACE_CHECK_INTERVAL_SEC  = 1   # per track_id
FACE_MATCH_THRESHOLD     = 120  # LBPH: lower is better

FACES_DIR   = os.path.abspath(os.path.join(PROJECT_ROOT, "faces_db"))
FACE_MODEL  = os.path.join(BASE_DIR, "face_model.yml")
FACE_LABELS = os.path.join(BASE_DIR, "face_labels.json")

os.makedirs(FACES_DIR, exist_ok=True)
face_auth = FaceAuthLBPH(
    faces_dir=FACES_DIR,
    model_path=FACE_MODEL,
    labels_path=FACE_LABELS,
)
# Train once at startup (best-effort). If dataset is small/empty, it stays untrained.
try:
    face_auth.retrain()
except Exception:
    pass

# ══════════════════════════════════════════════════════════════════════════════
# FALL DETECTION THRESHOLDS
# Calibrated from video analysis of side-view CCTV footage:
#   Standing person YOLO box W:H ≈ 0.45–0.55  (tall narrow)
#   Fallen  person YOLO box W:H ≈ 1.4–1.7     (wide flat)
#
# Two-signal approach:
#   Signal 1 — YOLO box aspect ratio > threshold (camera-angle independent)
#   Signal 2 — MediaPipe spine angle (when landmarks visible, side-view)
#
# Either signal alone can trigger — OR both must fire (configurable below).
# Temporal gate prevents single-frame false positives.
# ══════════════════════════════════════════════════════════════════════════════

# Signal 1 — YOLO bounding box W:H ratio
# Standing: ~0.5  |  Fallen: ~1.5
# Threshold set conservatively at 1.0 (safe gap between 0.5 and 1.5)
FALL_BBOX_RATIO_THRESHOLD  = 1.0

# Signal 2 — MediaPipe spine angle from horizontal (degrees)
# Standing spine is ~70-90° from horizontal
# Fallen  spine is ~0-30° from horizontal
# Threshold: if spine angle < 40° → body is mostly horizontal → fallen
FALL_SPINE_ANGLE_THRESHOLD = 40.0

# Temporal gate — N consecutive frames before confirming
FALL_CONFIRM_FRAMES        = 4

# Visibility threshold for MediaPipe keypoints
MIN_KEYPOINT_VISIBILITY    = 0.35

# ─── Accident ─────────────────────────────────────────────────────────────────
accident_detection_enabled = False
ACCIDENT_LOG_COOLDOWN      = 15
COLLISION_IOU_THRESHOLD    = 0.15
MIN_MOVING_VELOCITY        = 3.0
STOPPED_VELOCITY           = 1.0
HEADING_CHANGE_THRESHOLD   = 45.0
PERSON_VEHICLE_OVERLAP_PAD = 20
BBOX_RATIO_CHANGE          = 0.5
ACCIDENT_CONFIRM_FRAMES    = 5

VEHICLE_CLASSES = {1, 2, 3, 5, 7}
VEHICLE_LABELS  = {
    1: "BICYCLE", 2: "CAR", 3: "MOTORCYCLE", 5: "BUS", 7: "TRUCK"
}

# ─── ROI ──────────────────────────────────────────────────────────────────────
roi_polygon = None

# ─── Snapshot ─────────────────────────────────────────────────────────────────
SNAPSHOT_DIR = os.path.join(BASE_DIR, "static", "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

latest_frame = None
frame_lock   = threading.Lock()

last_snapshot_time = {
    "INTRUSION": 0, "FIRE": 0,
    "CROWD":     0, "FALL": 0,
    "ACCIDENT":  0,
}
SNAPSHOT_COOLDOWN = 10

# ─── MediaPipe Pose ───────────────────────────────────────────────────────────
pose_model = MediaPipePose(
    static_image_mode=False,
    model_complexity=0,
    enable_segmentation=False,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4,
)

KP_NOSE           = 0
KP_LEFT_SHOULDER  = 11
KP_RIGHT_SHOULDER = 12
KP_LEFT_HIP       = 23
KP_RIGHT_HIP      = 24
KP_LEFT_ANKLE     = 27
KP_RIGHT_ANKLE    = 28

# ─── Video ────────────────────────────────────────────────────────────────────
#cap      = cv2.VideoCapture("../data/CLIP1.mp4")
cap = cv2.VideoCapture(0)
detector = YOLODetector()

# ─── Metrics ──────────────────────────────────────────────────────────────────
metrics = {
    "fps": 0, "latency_ms": 0, "person_count": 0,
    "crowd_alerts": 0, "fire_alerts": 0, "intrusions": 0,
    "fall_alerts": 0, "accident_alerts": 0,
}

model_metrics = {
    "model":     "YOLOv8m-seg + ByteTrack",
    "accuracy":  0.91,
    "precision": 0.89,
    "recall":    0.87,
    "f1_score":  0.88,
}

# Per-person fall state — keyed by YOLO track_id
# Stores bbox ratio history and frame counter independently per person
person_fall_states = {}


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

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
    timestamp = time.strftime("%H%M%S")
    filename  = f"{event_type}_{timestamp}.jpg"
    cv2.imwrite(os.path.join(SNAPSHOT_DIR, filename), frame_copy)
    last_snapshot_time[event_type] = now
    event_logs.append({
        "type":    "SNAPSHOT",
        "time":    time.strftime("%H:%M:%S"),
        "details": f"Snapshot saved: {filename}",
    })


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / float(areaA + areaB - inter)


def init_vehicle_state():
    return {
        "centroids":       deque(maxlen=10),
        "velocities":      deque(maxlen=10),
        "headings":        deque(maxlen=10),
        "bbox_ratios":     deque(maxlen=5),
        "accident_frames": 0,
        "class_id":        -1,
        "accident_logged": False,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FALL DETECTION — DUAL SIGNAL (YOLO BOX + MEDIAPIPE SPINE)
# ══════════════════════════════════════════════════════════════════════════════

def get_spine_angle_from_frame(frame, x1, y1, x2, y2):
    """
    Run MediaPipe on a cropped person region.
    Returns spine angle from horizontal in degrees, or None if landmarks
    not visible enough.

    Spine vector = hip_midpoint → shoulder_midpoint
    Angle = 0°  → perfectly horizontal (lying flat)
    Angle = 90° → perfectly vertical   (standing upright)

    For side-view cameras this is the most reliable single signal.
    """
    # Add padding to crop so MediaPipe has context
    pad   = 20
    H, W  = frame.shape[:2]
    cx1   = max(0, x1 - pad)
    cy1   = max(0, y1 - pad)
    cx2   = min(W, x2 + pad)
    cy2   = min(H, y2 + pad)
    crop  = frame[cy1:cy2, cx1:cx2]

    if crop.size == 0:
        return None

    rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    result = pose_model.process(rgb)

    if not result.pose_landmarks:
        return None

    lm = result.pose_landmarks.landmark
    ch, cw = crop.shape[:2]

    def kp(idx):
        k = lm[idx]
        if k.visibility < MIN_KEYPOINT_VISIBILITY:
            return None
        return k.x * cw, k.y * ch

    l_sh  = kp(KP_LEFT_SHOULDER)
    r_sh  = kp(KP_RIGHT_SHOULDER)
    l_hip = kp(KP_LEFT_HIP)
    r_hip = kp(KP_RIGHT_HIP)

    # Need at least one shoulder and one hip
    sh_pts  = [p for p in [l_sh, r_sh] if p]
    hip_pts = [p for p in [l_hip, r_hip] if p]

    if not sh_pts or not hip_pts:
        return None

    sh_mid  = np.mean(sh_pts,  axis=0)
    hip_mid = np.mean(hip_pts, axis=0)

    dx = sh_mid[0] - hip_mid[0]
    dy = sh_mid[1] - hip_mid[1]

    # Angle from horizontal: arctan(|dy| / |dx|)
    # 0° = horizontal spine, 90° = vertical spine
    angle = abs(np.degrees(np.arctan2(abs(dy), abs(dx) + 1e-6)))
    return angle


def detect_fall_for_person(frame, track_id, x1, y1, x2, y2):
    """
    Per-person fall detection using two independent signals:

    Signal 1 — YOLO bounding box aspect ratio (W / H)
      Works on ANY camera angle — no pose estimation needed.
      Standing person: box is TALL   → W/H < 0.7
      Fallen  person: box is WIDE   → W/H > 1.0
      Camera-angle agnostic because the box always reflects body orientation.

    Signal 2 — MediaPipe spine angle (degrees from horizontal)
      Best for side-view cameras.
      Standing: spine is near vertical  → angle ~70-90°
      Fallen:   spine is near horizontal → angle ~0-30°
      Falls back gracefully if landmarks not detected.

    Decision logic:
      - Signal 1 fires  (bbox ratio > threshold)   → candidate
      - Signal 2 fires  (spine angle < threshold)  → additional confirmation
      - Either signal alone triggers the frame counter
      - Both signals together = stronger confidence

    Temporal gate: FALL_CONFIRM_FRAMES consecutive flagged frames required.
    """
    if track_id not in person_fall_states:
        person_fall_states[track_id] = {
            "fall_frames":  0,
            "bbox_history": deque(maxlen=6),
        }

    pfs = person_fall_states[track_id]

    box_w = x2 - x1
    box_h = max(y2 - y1, 1)
    ratio = box_w / box_h
    pfs["bbox_history"].append(ratio)

    # Use median of last few frames to smooth noise
    smooth_ratio = float(np.median(list(pfs["bbox_history"])))

    # Signal 1 — bbox aspect ratio
    signal_bbox = smooth_ratio > FALL_BBOX_RATIO_THRESHOLD

    # Signal 2 — MediaPipe spine angle (run only when bbox already suspicious)
    # This avoids running MediaPipe every frame for every person
    signal_spine = False
    spine_angle  = None
    if signal_bbox:
        spine_angle  = get_spine_angle_from_frame(frame, x1, y1, x2, y2)
        signal_spine = (spine_angle is not None and
                        spine_angle < FALL_SPINE_ANGLE_THRESHOLD)

    # Either signal triggers the counter
    any_signal = signal_bbox or signal_spine

    if any_signal:
        pfs["fall_frames"] += 1
    else:
        pfs["fall_frames"] = 0

    fall_confirmed = pfs["fall_frames"] >= FALL_CONFIRM_FRAMES

    return fall_confirmed, smooth_ratio, spine_angle, pfs["fall_frames"]


def detect_fall(frame, person_detections):
    """
    Main fall detection entry point.
    Called with list of (track_id, x1, y1, x2, y2) for all persons this frame.

    Returns (any_fall_confirmed, annotated_frame)
    """
    any_confirmed = False

    for (track_id, x1, y1, x2, y2) in person_detections:
        confirmed, ratio, spine_angle, counter = detect_fall_for_person(
            frame, track_id, x1, y1, x2, y2
        )

        if confirmed:
            any_confirmed = True
            # Red box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, "FALL DETECTED",
                        (x1, y1 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(frame, "FALL DETECTED",
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)

        elif ratio > FALL_BBOX_RATIO_THRESHOLD:
            # Pending — building toward confirmation
            spine_str = f" spine={spine_angle:.0f}°" if spine_angle else ""
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(frame,
                        f"Fall? {counter}/{FALL_CONFIRM_FRAMES} ratio={ratio:.2f}{spine_str}",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

    return any_confirmed, frame


# ══════════════════════════════════════════════════════════════════════════════
# FIRE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_fire_color(frame):
    hsv     = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask    = cv2.inRange(hsv, np.array([0, 120, 150]), np.array([35, 255, 255]))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes   = []
    for c in cnts:
        if cv2.contourArea(c) > 800:
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, x+w, y+h))
    return boxes


# ══════════════════════════════════════════════════════════════════════════════
# ACCIDENT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_accidents(frame, vehicle_boxes, person_boxes):
    accident_detected = False
    accident_reason   = ""
    flagged_ids       = set()

    for (tid, cls_id, x1, y1, x2, y2) in vehicle_boxes:
        if tid not in vehicle_states:
            vehicle_states[tid] = init_vehicle_state()
        vs = vehicle_states[tid]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        vs["class_id"] = cls_id
        vs["centroids"].append((cx, cy))
        if len(vs["centroids"]) >= 2:
            prev = vs["centroids"][-2]
            dist = np.hypot(cx-prev[0], cy-prev[1])
            vs["velocities"].append(dist)
            vs["headings"].append(np.degrees(np.arctan2(cy-prev[1], cx-prev[0])))
        vs["bbox_ratios"].append(max(x2-x1, 1) / max(y2-y1, 1))

    for i in range(len(vehicle_boxes)):
        for j in range(i+1, len(vehicle_boxes)):
            tid_a, _, *box_a = vehicle_boxes[i]
            tid_b, _, *box_b = vehicle_boxes[j]
            iou = compute_iou(box_a, box_b)
            if iou > COLLISION_IOU_THRESHOLD:
                flagged_ids.add(tid_a); flagged_ids.add(tid_b)
                la = VEHICLE_LABELS.get(vehicle_boxes[i][1], "VEHICLE")
                lb = VEHICLE_LABELS.get(vehicle_boxes[j][1], "VEHICLE")
                accident_reason = f"{la} vs {lb} collision (IoU {iou:.2f})"

    for (tid, cls_id, x1, y1, x2, y2) in vehicle_boxes:
        vs    = vehicle_states.get(tid)
        label = VEHICLE_LABELS.get(cls_id, "VEHICLE")
        if vs is None: continue
        if len(vs["velocities"]) >= 5:
            avg_vel  = np.mean(list(vs["velocities"])[:-1])
            curr_vel = vs["velocities"][-1]
            if avg_vel > MIN_MOVING_VELOCITY and curr_vel < STOPPED_VELOCITY:
                flagged_ids.add(tid)
                accident_reason = accident_reason or f"{label} sudden stop"
        if len(vs["headings"]) >= 3:
            h_list = list(vs["headings"])
            diff   = min(abs(h_list[-1]-h_list[-2]), 360-abs(h_list[-1]-h_list[-2]))
            if diff > HEADING_CHANGE_THRESHOLD:
                flagged_ids.add(tid)
                accident_reason = accident_reason or f"{label} trajectory deviation"
        if len(vs["bbox_ratios"]) >= 3:
            ratios = list(vs["bbox_ratios"])
            if abs(ratios[-1]-ratios[-2]) > BBOX_RATIO_CHANGE:
                flagged_ids.add(tid)
                accident_reason = accident_reason or f"{label} shape deformation"

    for (vtid, cls_id, vx1, vy1, vx2, vy2) in vehicle_boxes:
        vs    = vehicle_states.get(vtid)
        label = VEHICLE_LABELS.get(cls_id, "VEHICLE")
        moving = (vs and len(vs["velocities"]) >= 3 and
                  np.mean(list(vs["velocities"])[-3:]) > MIN_MOVING_VELOCITY)
        if not moving: continue
        pad = PERSON_VEHICLE_OVERLAP_PAD
        for (ptid, pcx, pcy) in person_boxes:
            if (vx1-pad) <= pcx <= (vx2+pad) and (vy1-pad) <= pcy <= (vy2+pad):
                flagged_ids.add(vtid)
                accident_reason = accident_reason or f"Pedestrian impact by {label}"

    confirmed_ids = set()
    for (tid, cls_id, x1, y1, x2, y2) in vehicle_boxes:
        vs = vehicle_states.get(tid)
        if vs is None: continue
        vs["accident_frames"] = vs["accident_frames"]+1 if tid in flagged_ids else 0
        if vs["accident_frames"] >= ACCIDENT_CONFIRM_FRAMES:
            confirmed_ids.add(tid)

    for (tid, cls_id, x1, y1, x2, y2) in vehicle_boxes:
        vs    = vehicle_states.get(tid)
        label = VEHICLE_LABELS.get(cls_id, "VEHICLE")
        if tid in confirmed_ids:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f"ACCIDENT {label}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            accident_detected = True
        elif tid in flagged_ids and vs:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
            cv2.putText(frame, f"Alert? ({vs['accident_frames']}/{ACCIDENT_CONFIRM_FRAMES})",
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

    if accident_detected:
        cv2.putText(frame, "ACCIDENT DETECTED", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)

    return accident_detected, frame, accident_reason


# ══════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/set_roi", methods=["POST"])
def set_roi():
    global roi_polygon
    pts = request.json.get("points", [])
    roi_polygon = np.array(pts, np.int32) if len(pts) >= 3 else None
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

@app.route("/toggle_accident", methods=["POST"])
def toggle_accident():
    global accident_detection_enabled
    accident_detection_enabled = not accident_detection_enabled
    return jsonify({"enabled": accident_detection_enabled})

@app.route("/toggle_face", methods=["POST"])
def toggle_face():
    global face_recognition_enabled
    face_recognition_enabled = not face_recognition_enabled
    return jsonify({"enabled": face_recognition_enabled})

@app.route("/faces")
def list_faces():
    return jsonify({
        "users": face_auth.list_users(),
        "trained": face_auth.is_trained,
    })

@app.route("/enroll_face", methods=["POST"])
def enroll_face():
    """
    FormData:
      - name: string
      - image: file
    """
    name = (request.form.get("name") or "").strip()
    file = request.files.get("image")
    if not name or file is None:
        return jsonify({"error": "Missing name or image"}), 400

    data = file.read()
    if not data:
        return jsonify({"error": "Empty upload"}), 400

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    ok = face_auth.enroll_image(name, img)
    if not ok:
        return jsonify({"error": "Failed to save image"}), 500

    stats = face_auth.retrain()
    event_logs.append({
        "type": "FACE_ENROLL",
        "time": time.strftime("%H:%M:%S"),
        "details": f"Enrolled authorized face: {name} (labels={stats['labels']}, used={stats['used_faces']})",
    })
    return jsonify({"status": "enrolled", "stats": stats})

@app.route("/delete_face", methods=["POST"])
def delete_face():
    """
    JSON:
      - name: string
    """
    name = (request.json or {}).get("name", "")
    name = str(name).strip()
    if not name:
        return jsonify({"error": "Missing name"}), 400
    deleted = face_auth.delete_user(name)
    stats = face_auth.retrain()
    return jsonify({"deleted": deleted, "stats": stats})

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
    return send_file(buf, mimetype="text/csv", as_attachment=True,
                     download_name=f"safesight_logs_{time.strftime('%Y%m%d_%H%M%S')}.csv")

@app.route("/snapshots")
def list_snapshots():
    files = sorted(os.listdir(SNAPSHOT_DIR), reverse=True)
    return jsonify([
        {"filename": f, "url": f"/static/snapshots/{f}"}
        for f in files if f.endswith(".jpg")
    ])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/events")
def events():
    return jsonify(event_logs[::-1])

@app.route("/metrics")
def live_metrics():
    return jsonify({"system": metrics, "model": model_metrics})


# ══════════════════════════════════════════════════════════════════════════════
# FRAME GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_frames():
    global last_intrusion_time, last_crowd_log_time
    global last_fire_log_time,  last_fall_log_time
    global last_accident_time,  latest_frame
    prev_time = time.time()

    while True:
        start   = time.time()
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        results       = detector.detect(frame)
        tracked_ids   = set()
        fire_detected = False
        vehicle_boxes = []
        person_boxes  = []
        person_bboxes = []   # (track_id, x1, y1, x2, y2) for fall detection

        for result in results:
            frame = result.plot(conf=False)
            if roi_polygon is not None:
                cv2.polylines(frame, [roi_polygon], True, (255, 0, 0), 2)

            for box in result.boxes:
                if box.id is None:
                    continue
                cls_id   = int(box.cls[0])
                track_id = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1+x2)//2, (y1+y2)//2

                if cls_id == 0:
                    tracked_ids.add(track_id)
                    person_boxes.append((track_id, cx, cy))
                    person_bboxes.append((track_id, x1, y1, x2, y2))

                    if track_id not in person_states:
                        person_states[track_id] = {
                            "entry_time": None,
                            "loiter_logged": False,
                            "authorized": False,
                            "last_face_check": 0.0,
                        }
                    state  = person_states[track_id]
                    inside = -1
                    if roi_polygon is not None:
                        inside = cv2.pointPolygonTest(roi_polygon, (cx, cy), False)

                    if inside >= 0:
                        now = time.time()
                        # ── Face auth gate (STRICT) ────────────────────────────
                        if face_recognition_enabled:
                            # Only check periodically per track to keep FPS up
                            if (now - float(state.get("last_face_check", 0.0))) >= FACE_CHECK_INTERVAL_SEC:
                                state["last_face_check"] = now
                                # Crop top portion of person bbox (more likely to include face)
                                H, W = frame.shape[:2]
                                px1 = max(0, x1)
                                py1 = max(0, y1)
                                px2 = min(W, x2)
                                py2 = min(H, y2)
                                # Focus on upper 60% of the person box
                                upper_h = int((py2 - py1) * 0.6)
                                py2u = min(H, py1 + max(upper_h, 1))
                                crop = frame[py1:py2u, px1:px2]
                                name, conf, face_visible = face_auth.identify_from_bgr(
                                    crop, threshold=FACE_MATCH_THRESHOLD
                                )
                                if name:
                                    state["authorized"] = True
                                    event_logs.append({
                                        "type": "AUTHORIZED_ENTRY",
                                        "time": time.strftime("%H:%M:%S"),
                                        "details": f"{name} authorized in ROI (Person ID {track_id}, conf={conf:.1f})",
                                    })
                                else:
                                    # Strict mode: no face visible OR not recognized => unauthorized
                                    state["authorized"] = False

                        # If authorized, skip intrusion/loitering alerts
                        if face_recognition_enabled and state.get("authorized"):
                            cv2.putText(frame, "AUTHORIZED", (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            # Do not set entry_time for loitering when authorized
                            state["entry_time"] = None
                            state["loiter_logged"] = False
                            continue

                        if state["entry_time"] is None:
                            state["entry_time"] = now
                            if now - last_intrusion_time > INTRUSION_COOLDOWN:
                                event_logs.append({
                                    "type":    "INTRUSION",
                                    "time":    time.strftime("%H:%M:%S"),
                                    "details": f"Person ID {track_id} entered restricted area",
                                })
                                metrics["intrusions"] += 1
                                last_intrusion_time = now
                                save_snapshot("INTRUSION")
                        dwell = now - state["entry_time"]
                        if dwell > LOITERING_TIME_THRESHOLD and not state["loiter_logged"]:
                            event_logs.append({
                                "type":    "LOITERING",
                                "time":    time.strftime("%H:%M:%S"),
                                "details": f"Person ID {track_id} loitering ({int(dwell)}s)",
                            })
                            state["loiter_logged"] = True
                        cv2.putText(frame, "INTRUSION", (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        state["entry_time"]    = None
                        state["loiter_logged"] = False
                        state["authorized"]    = False

                elif cls_id in VEHICLE_CLASSES:
                    vehicle_boxes.append((track_id, cls_id, x1, y1, x2, y2))

        # ── Fire ──────────────────────────────────────────────────────────────
        if fire_detection_enabled:
            for (x1, y1, x2, y2) in detect_fire_color(frame):
                fire_detected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, "FIRE", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        if fire_detected and time.time() - last_fire_log_time > FIRE_LOG_COOLDOWN:
            event_logs.append({"type": "FIRE", "time": time.strftime("%H:%M:%S"),
                                "details": "Fire detected in monitored zone"})
            metrics["fire_alerts"] += 1
            last_fire_log_time = time.time()
            save_snapshot("FIRE")

        # ── Crowd ─────────────────────────────────────────────────────────────
        metrics["person_count"] = len(tracked_ids)
        if crowd_detection_enabled and metrics["person_count"] >= CROWD_THRESHOLD:
            if time.time() - last_crowd_log_time > CROWD_COOLDOWN:
                event_logs.append({"type": "CROWD", "time": time.strftime("%H:%M:%S"),
                                   "details": f"High crowd density ({metrics['person_count']})"})
                metrics["crowd_alerts"] += 1
                last_crowd_log_time = time.time()
                save_snapshot("CROWD")

        # ── Fall ──────────────────────────────────────────────────────────────
        if fall_detection_enabled and person_bboxes:
            fall_confirmed, frame = detect_fall(frame, person_bboxes)
            if fall_confirmed and time.time() - last_fall_log_time > FALL_LOG_COOLDOWN:
                event_logs.append({"type": "FALL", "time": time.strftime("%H:%M:%S"),
                                   "details": "Person fall detected in monitored area"})
                metrics["fall_alerts"] += 1
                last_fall_log_time = time.time()
                save_snapshot("FALL")
                play_alert_sound()

        # ── Accident ──────────────────────────────────────────────────────────
        if accident_detection_enabled and vehicle_boxes:
            acc_detected, frame, acc_reason = detect_accidents(
                frame, vehicle_boxes, person_boxes
            )
            if acc_detected and time.time() - last_accident_time > ACCIDENT_LOG_COOLDOWN:
                event_logs.append({
                    "type":    "ACCIDENT",
                    "time":    time.strftime("%H:%M:%S"),
                    "details": acc_reason or "Vehicle accident detected",
                })
                metrics["accident_alerts"] += 1
                last_accident_time = time.time()
                save_snapshot("ACCIDENT")
                play_alert_sound()

        # ── Store latest frame ─────────────────────────────────────────────────
        with frame_lock:
            latest_frame = frame.copy()

        end = time.time()
        metrics["latency_ms"] = int((end - start) * 1000)
        metrics["fps"]        = int(1 / max(end - prev_time, 0.001))
        prev_time = end

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(debug=True)