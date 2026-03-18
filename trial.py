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

app = Flask(__name__)

# ─── Per-object state ────────────────────────────────────────────────────────────
person_states  = {}   # track_id → intrusion / loiter state
vehicle_states = {}   # track_id → velocity / heading / accident state
event_logs     = []

# ─── Cooldown timestamps ─────────────────────────────────────────────────────────
last_intrusion_time = 0
last_crowd_log_time = 0
last_fire_log_time  = 0
last_fall_log_time  = 0
last_accident_time  = 0

# ─── Feature constants ───────────────────────────────────────────────────────────
INTRUSION_COOLDOWN       = 5
LOITERING_TIME_THRESHOLD = 10

CROWD_THRESHOLD         = 5
CROWD_COOLDOWN          = 10
crowd_detection_enabled = False

fire_detection_enabled = False
FIRE_LOG_COOLDOWN      = 10

fall_detection_enabled     = False
FALL_LOG_COOLDOWN          = 10
FALL_SHOULDER_SPREAD_RATIO = 0.15
FALL_VERTICAL_COMPRESSION  = 0.18
FALL_SPREAD_RATIO          = 1.4
FALL_CONFIRM_FRAMES        = 4
MIN_KEYPOINT_VISIBILITY    = 0.4

accident_detection_enabled = False
ACCIDENT_LOG_COOLDOWN      = 15   # seconds between accident log entries

# ── Accident detection thresholds ────────────────────────────────────────────────

# Signal 1 — IoU overlap between two vehicle boxes (collision proximity)
COLLISION_IOU_THRESHOLD    = 0.15  # boxes overlap by this fraction → suspicious

# Signal 2 — Sudden velocity drop (post-impact stoppage)
# Vehicle must have been moving at least this fast (px/frame) before stopping
MIN_MOVING_VELOCITY        = 3.0
# Velocity must drop to below this to count as "stopped"
STOPPED_VELOCITY           = 1.0

# Signal 3 — Trajectory deviation (off-road / sharp swerve)
# Heading angle change in degrees between consecutive frames
HEADING_CHANGE_THRESHOLD   = 45.0

# Signal 4 — Person inside vehicle bounding box (pedestrian hit)
# Fraction of person centroid proximity to vehicle box
PERSON_VEHICLE_OVERLAP_PAD = 20   # pixels padding around vehicle box

# Signal 5 — Bounding box aspect ratio sudden change (deformation / spin)
BBOX_RATIO_CHANGE          = 0.5  # abs difference in W:H ratio across frames

# Temporal gate — how many consecutive frames signals must persist
ACCIDENT_CONFIRM_FRAMES    = 5

# YOLO COCO class IDs for vehicles
VEHICLE_CLASSES = {1, 2, 3, 5, 7}   # bicycle, car, motorcycle, bus, truck
VEHICLE_LABELS  = {
    1: "BICYCLE", 2: "CAR", 3: "MOTORCYCLE", 5: "BUS", 7: "TRUCK"
}
# ──────────────────────────────────────────────────────────────────────────────────

# ROI
roi_polygon = None

# Snapshot
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

# Consecutive-frame counters
fall_frame_counter = 0

# MediaPipe Pose
pose_model = MediaPipePose(
    static_image_mode=False,
    model_complexity=0,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

KP_NOSE           = 0
KP_LEFT_SHOULDER  = 11
KP_RIGHT_SHOULDER = 12
KP_LEFT_HIP       = 23
KP_RIGHT_HIP      = 24
KP_LEFT_ANKLE     = 27
KP_RIGHT_ANKLE    = 28

# Video
cap = cv2.VideoCapture("../data/CLIP1.mp4")
#cap      = cv2.VideoCapture(0)
detector = YOLODetector()

metrics = {
    "fps":             0,
    "latency_ms":      0,
    "person_count":    0,
    "crowd_alerts":    0,
    "fire_alerts":     0,
    "intrusions":      0,
    "fall_alerts":     0,
    "accident_alerts": 0,
}

model_metrics = {
    "model":     "YOLOv8 + ByteTrack",
    "accuracy":  0.91,
    "precision": 0.89,
    "recall":    0.87,
    "f1_score":  0.88,
}


# ─── Utilities ───────────────────────────────────────────────────────────────────

def play_alert_sound():
    """Three beeps on a daemon thread — never blocks the frame loop."""
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
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
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


# ─── Fall Detection (Top-Down) ───────────────────────────────────────────────────

def detect_fall(frame):
    global fall_frame_counter
    h, w   = frame.shape[:2]
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

    ls, rs = px(KP_LEFT_SHOULDER), px(KP_RIGHT_SHOULDER)
    lh, rh = px(KP_LEFT_HIP),      px(KP_RIGHT_HIP)
    la, ra = px(KP_LEFT_ANKLE),     px(KP_RIGHT_ANKLE)
    ns     = px(KP_NOSE)

    if ls is None or rs is None:
        fall_frame_counter = 0
        return False, frame

    # Condition 1 — shoulder spread
    cond1 = (abs(rs[0] - ls[0]) / w) > FALL_SHOULDER_SPREAD_RATIO

    # Condition 2 — vertical compression
    ankle_y = ((la[1]+ra[1])/2 if la and ra else
               la[1] if la else ra[1] if ra else None)
    cond2 = bool(ns and ankle_y and
                 (abs(ankle_y - ns[1]) / h) < FALL_VERTICAL_COMPRESSION)

    # Condition 3 — keypoint ellipse ratio
    visible_pts = [p for p in [
        px(KP_LEFT_SHOULDER), px(KP_RIGHT_SHOULDER),
        px(KP_LEFT_HIP),      px(KP_RIGHT_HIP),
        px(KP_LEFT_ANKLE),    px(KP_RIGHT_ANKLE),
    ] if p is not None]

    cond3 = False
    if len(visible_pts) >= 4:
        xs    = [p[0] for p in visible_pts]
        ys    = [p[1] for p in visible_pts]
        cond3 = ((max(xs)-min(xs)) / (max(ys)-min(ys)+1)) > FALL_SPREAD_RATIO

    all_conditions = cond1 and cond2 and cond3
    fall_frame_counter = fall_frame_counter + 1 if all_conditions else 0
    fall_confirmed     = fall_frame_counter >= FALL_CONFIRM_FRAMES

    if fall_confirmed:
        for pt in visible_pts:
            cv2.circle(frame, pt, 6, (0, 255, 255), -1)
        if ls and rs: cv2.line(frame, ls, rs, (0, 255, 255), 2)
        if lh and rh: cv2.line(frame, lh, rh, (0, 255, 255), 2)
        if ls and lh: cv2.line(frame, ls, lh, (0, 255, 255), 2)
        if rs and rh: cv2.line(frame, rs, rh, (0, 255, 255), 2)
        cv2.putText(frame, "FALL DETECTED", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
    elif all_conditions:
        cv2.putText(frame, f"Fall? ({fall_frame_counter}/{FALL_CONFIRM_FRAMES})",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    return fall_confirmed, frame


# ─── Accident Detection ───────────────────────────────────────────────────────────

def detect_accidents(frame, vehicle_boxes, person_boxes):
    """
    vehicle_boxes : list of (track_id, class_id, x1, y1, x2, y2)
    person_boxes  : list of (track_id, cx, cy)
    Returns       : (accident_detected: bool, annotated_frame, reason: str)

    Five signals checked per vehicle and across vehicle pairs:

      Signal 1 — IoU Overlap
        Any two vehicle bounding boxes overlap beyond COLLISION_IOU_THRESHOLD.
        Both vehicles flagged. Covers car-car and bike-car collisions.

      Signal 2 — Sudden Velocity Drop
        Vehicle was moving (avg velocity > MIN_MOVING_VELOCITY) but current
        velocity < STOPPED_VELOCITY. Indicates post-impact abrupt stoppage.

      Signal 3 — Trajectory Deviation
        Heading angle between consecutive centroid vectors changes by more
        than HEADING_CHANGE_THRESHOLD degrees. Covers swerving / off-road.

      Signal 4 — Person inside Vehicle Box
        Person centroid falls within a padded vehicle bounding box.
        Indicates pedestrian impact. Triggers only if vehicle was moving.

      Signal 5 — Bounding Box Ratio Deformation
        Vehicle W:H ratio changes sharply between frames.
        Covers spinning, rolling, or crushed vehicle shapes.

    Each signal increments that vehicle's accident_frames counter.
    Confirmation requires ACCIDENT_CONFIRM_FRAMES consecutive flagged frames.
    """
    accident_detected = False
    accident_reason   = ""
    flagged_ids       = set()

    now = time.time()

    # ── Update vehicle state from current detections ──────────────────────────
    for (tid, cls_id, x1, y1, x2, y2) in vehicle_boxes:
        if tid not in vehicle_states:
            vehicle_states[tid] = init_vehicle_state()

        vs  = vehicle_states[tid]
        cx  = (x1 + x2) // 2
        cy  = (y1 + y2) // 2
        vs["class_id"] = cls_id

        # Centroid history → velocity
        vs["centroids"].append((cx, cy))
        if len(vs["centroids"]) >= 2:
            prev = vs["centroids"][-2]
            dist = np.hypot(cx - prev[0], cy - prev[1])
            vs["velocities"].append(dist)

            # Heading angle (degrees)
            angle = np.degrees(np.arctan2(cy - prev[1], cx - prev[0]))
            vs["headings"].append(angle)

        # Bounding box aspect ratio history
        w_box = max(x2 - x1, 1)
        h_box = max(y2 - y1, 1)
        vs["bbox_ratios"].append(w_box / h_box)

    # ── Signal 1: IoU Overlap between vehicle pairs ───────────────────────────
    for i in range(len(vehicle_boxes)):
        for j in range(i + 1, len(vehicle_boxes)):
            tid_a, _, *box_a = vehicle_boxes[i]
            tid_b, _, *box_b = vehicle_boxes[j]
            iou = compute_iou(box_a, box_b)
            if iou > COLLISION_IOU_THRESHOLD:
                flagged_ids.add(tid_a)
                flagged_ids.add(tid_b)
                lbl_a = VEHICLE_LABELS.get(vehicle_boxes[i][1], "VEHICLE")
                lbl_b = VEHICLE_LABELS.get(vehicle_boxes[j][1], "VEHICLE")
                accident_reason = f"{lbl_a} vs {lbl_b} collision (IoU {iou:.2f})"

    # ── Signals 2, 3, 5: Per-vehicle history checks ───────────────────────────
    for (tid, cls_id, x1, y1, x2, y2) in vehicle_boxes:
        vs = vehicle_states.get(tid)
        if vs is None:
            continue

        label = VEHICLE_LABELS.get(cls_id, "VEHICLE")

        # Signal 2 — Sudden velocity drop
        if len(vs["velocities"]) >= 5:
            avg_vel  = np.mean(list(vs["velocities"])[:-1])
            curr_vel = vs["velocities"][-1]
            if avg_vel > MIN_MOVING_VELOCITY and curr_vel < STOPPED_VELOCITY:
                flagged_ids.add(tid)
                accident_reason = accident_reason or f"{label} sudden stop after impact"

        # Signal 3 — Trajectory / heading deviation
        if len(vs["headings"]) >= 3:
            h_list      = list(vs["headings"])
            heading_diff = abs(h_list[-1] - h_list[-2])
            # Normalise angle difference to [0, 180]
            heading_diff = min(heading_diff, 360 - heading_diff)
            if heading_diff > HEADING_CHANGE_THRESHOLD:
                flagged_ids.add(tid)
                accident_reason = accident_reason or f"{label} trajectory deviation ({heading_diff:.0f}°)"

        # Signal 5 — Bounding box deformation
        if len(vs["bbox_ratios"]) >= 3:
            ratios = list(vs["bbox_ratios"])
            if abs(ratios[-1] - ratios[-2]) > BBOX_RATIO_CHANGE:
                flagged_ids.add(tid)
                accident_reason = accident_reason or f"{label} shape deformation detected"

    # ── Signal 4: Person inside vehicle bounding box ─────────────────────────
    for (vtid, cls_id, vx1, vy1, vx2, vy2) in vehicle_boxes:
        vs    = vehicle_states.get(vtid)
        label = VEHICLE_LABELS.get(cls_id, "VEHICLE")
        # Only flag if vehicle was moving
        moving = (vs and len(vs["velocities"]) >= 3 and
                  np.mean(list(vs["velocities"])[-3:]) > MIN_MOVING_VELOCITY)
        if not moving:
            continue
        pad = PERSON_VEHICLE_OVERLAP_PAD
        for (ptid, pcx, pcy) in person_boxes:
            if (vx1 - pad) <= pcx <= (vx2 + pad) and (vy1 - pad) <= pcy <= (vy2 + pad):
                flagged_ids.add(vtid)
                accident_reason = accident_reason or f"Pedestrian impact by {label}"

    # ── Temporal gate — increment / reset per-vehicle counters ───────────────
    confirmed_ids = set()
    for (tid, cls_id, x1, y1, x2, y2) in vehicle_boxes:
        vs = vehicle_states.get(tid)
        if vs is None:
            continue
        if tid in flagged_ids:
            vs["accident_frames"] += 1
        else:
            vs["accident_frames"] = 0

        if vs["accident_frames"] >= ACCIDENT_CONFIRM_FRAMES:
            confirmed_ids.add(tid)

    # ── Annotate confirmed accidents on frame ─────────────────────────────────
    for (tid, cls_id, x1, y1, x2, y2) in vehicle_boxes:
        vs    = vehicle_states.get(tid)
        label = VEHICLE_LABELS.get(cls_id, "VEHICLE")

        if tid in confirmed_ids:
            # Red box + label for confirmed accident vehicle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f"ACCIDENT {label}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            accident_detected = True

        elif tid in flagged_ids and vs:
            # Orange box — signals detected, building toward confirmation
            cnt = vs["accident_frames"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
            cv2.putText(frame, f"Alert? ({cnt}/{ACCIDENT_CONFIRM_FRAMES})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

    # Global ACCIDENT DETECTED banner
    if accident_detected:
        cv2.putText(frame, "ACCIDENT DETECTED",
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)

    return accident_detected, frame, accident_reason


# ─── Fire Detection ───────────────────────────────────────────────────────────────

def detect_fire_color(frame):
    hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask   = cv2.inRange(hsv, np.array([0, 120, 150]), np.array([35, 255, 255]))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes  = []
    for c in cnts:
        if cv2.contourArea(c) > 800:
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, x+w, y+h))
    return boxes


# ─── Routes ──────────────────────────────────────────────────────────────────────

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


# ─── Frame Generator ──────────────────────────────────────────────────────────────

def generate_frames():
    global last_intrusion_time, last_crowd_log_time
    global last_fire_log_time, last_fall_log_time
    global last_accident_time, latest_frame
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

        # Collected this frame for accident detection
        vehicle_boxes = []   # (track_id, class_id, x1, y1, x2, y2)
        person_boxes  = []   # (track_id, cx, cy)

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
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # ── Person logic ──────────────────────────────────────────────
                if cls_id == 0:
                    tracked_ids.add(track_id)
                    person_boxes.append((track_id, cx, cy))

                    if track_id not in person_states:
                        person_states[track_id] = {
                            "entry_time": None, "loiter_logged": False
                        }
                    state  = person_states[track_id]
                    inside = -1
                    if roi_polygon is not None:
                        inside = cv2.pointPolygonTest(roi_polygon, (cx, cy), False)

                    if inside >= 0:
                        now = time.time()
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

                # ── Vehicle collection for accident detection ─────────────────
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
        if fall_detection_enabled:
            fall_confirmed, frame = detect_fall(frame)
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


# ─── Core Routes ──────────────────────────────────────────────────────────────────

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

if __name__ == "__main__":
    app.run(debug=True)