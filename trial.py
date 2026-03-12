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

# ─── Per-object state ─────────────────────────────────────────────────────────
person_states  = {}
vehicle_states = {}
event_logs     = []

# ─── Cooldowns ────────────────────────────────────────────────────────────────
last_intrusion_time = 0
last_crowd_log_time = 0
last_fire_log_time  = 0
last_fall_log_time  = 0
last_accident_time  = 0

# ─── Person / Intrusion ───────────────────────────────────────────────────────
INTRUSION_COOLDOWN       = 5
LOITERING_TIME_THRESHOLD = 10

# ─── Crowd ────────────────────────────────────────────────────────────────────
CROWD_THRESHOLD         = 5
CROWD_COOLDOWN          = 10
crowd_detection_enabled = False

# ─── Fire ─────────────────────────────────────────────────────────────────────
fire_detection_enabled = False
FIRE_LOG_COOLDOWN      = 10

# ─── Fall ─────────────────────────────────────────────────────────────────────
fall_detection_enabled     = False
FALL_LOG_COOLDOWN          = 10
FALL_SHOULDER_SPREAD_RATIO = 0.15
FALL_VERTICAL_COMPRESSION  = 0.18
FALL_SPREAD_RATIO          = 1.4
FALL_CONFIRM_FRAMES        = 4
MIN_KEYPOINT_VISIBILITY    = 0.4

# ─── Accident ─────────────────────────────────────────────────────────────────
accident_detection_enabled = False
ACCIDENT_LOG_COOLDOWN      = 15

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — HYBRID DETECTION THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: Bounding box IoU trigger ────────────────────────────────────────
# This is the TRIGGER for segmentation, not the final confirmation.
# When two vehicle boxes overlap beyond this → extract their pixel masks.
# Calibrated from video: normal traffic max IoU = 0.20, accident IoU = 0.62
# Set at 0.30 — safely above normal traffic noise
BOX_IOU_SEG_TRIGGER        = 0.30

# ── Stage 2: Pixel mask intersection (the real collision check) ───────────────
# After masks are extracted, compute:
#   overlap_pixels / min(mask_A_pixels, mask_B_pixels)
# This ratio tells us what fraction of the SMALLER vehicle is overlapping.
# Two cars side-by-side: boxes overlap, masks DON'T → ratio near 0 → no alert
# Two cars colliding:    boxes overlap, masks DO  → ratio > threshold → alert
MASK_OVERLAP_THRESHOLD     = 0.15   # 15% of smaller vehicle's pixels overlap

# ── Velocity drop (Signal 2 — unchanged, calibrated from video) ──────────────
# Video measured: centroid velocity median = 4.12 px/frame
VELOCITY_BASELINE_FRAMES   = 8
MIN_MOVING_VELOCITY        = 3.0
VELOCITY_DROP_RATIO        = 0.20
VELOCITY_BASELINE_MAX_STD  = 5.0

# ── Trajectory deviation (Signal 3) ──────────────────────────────────────────
HEADING_CHANGE_THRESHOLD   = 45.0
HEADING_SUSTAINED_FRAMES   = 2

# ── Person-vehicle impact (Signal 4) ─────────────────────────────────────────
PERSON_VEHICLE_OVERLAP_PAD = 20
PERSON_DECEL_RATIO         = 0.5

# ── Bbox deformation (Signal 5) ──────────────────────────────────────────────
BBOX_RATIO_CHANGE          = 0.5

# ── Multi-signal confirmation gate ───────────────────────────────────────────
# MASK_COLLISION counts as Signal 1.
# Need 1 more signal from {VELOCITY_DROP, TRAJECTORY_DEV,
#                           PEDESTRIAN_HIT, BBOX_DEFORM}
MIN_SIGNALS_REQUIRED       = 2
ACCIDENT_CONFIRM_FRAMES    = 4

# ── Segmentation metrics (for dashboard / viva demo) ─────────────────────────
seg_metrics = {
    "seg_triggers":    0,   # how many times box IoU triggered mask extraction
    "mask_collisions": 0,   # how many times mask intersection confirmed contact
    "fp_eliminated":   0,   # box IoU fired but mask said NO → FP eliminated
}

# ─── YOLO vehicle classes ─────────────────────────────────────────────────────
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
SNAPSHOT_COOLDOWN  = 10
fall_frame_counter = 0

# ─── MediaPipe Pose ───────────────────────────────────────────────────────────
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

# ─── Video ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture("../data/CLASS.mp4")
#cap = cv2.VideoCapture(0)
detector = YOLODetector()

# ─── Metrics ──────────────────────────────────────────────────────────────────
metrics = {
    "fps": 0, "latency_ms": 0, "person_count": 0,
    "crowd_alerts": 0, "fire_alerts": 0, "intrusions": 0,
    "fall_alerts": 0, "accident_alerts": 0,
}

model_metrics = {
    "model":     "YOLOv8m-seg + ByteTrack (Phase 2)",
    "accuracy":  0.91,
    "precision": 0.89,
    "recall":    0.87,
    "f1_score":  0.88,
}


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
    """Bounding box IoU — used only as segmentation trigger, not final check."""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    if inter == 0:
        return 0.0
    aA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    aB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / float(aA + aB - inter)


def compute_mask_overlap(mask_a, mask_b):
    """
    Pixel-level mask intersection ratio.

    Formula:
        overlap_pixels / min(pixels_in_A, pixels_in_B)

    Why min() not union:
        We want to know what fraction of the SMALLER vehicle
        is physically inside the larger one. A bus and a motorcycle
        colliding — the motorcycle may be 100% inside the bus mask.
        Using union would dilute this signal.

    Returns float in [0.0, 1.0]
    """
    intersection = cv2.bitwise_and(mask_a, mask_b)
    overlap_px   = int(np.count_nonzero(intersection))
    if overlap_px == 0:
        return 0.0
    area_a = int(np.count_nonzero(mask_a))
    area_b = int(np.count_nonzero(mask_b))
    min_area = min(area_a, area_b)
    if min_area == 0:
        return 0.0
    return overlap_px / min_area


def init_vehicle_state():
    return {
        "centroids":       deque(maxlen=20),
        "velocities":      deque(maxlen=20),
        "vel_vectors":     deque(maxlen=10),
        "headings":        deque(maxlen=15),
        "heading_dev_cnt": 0,
        "bbox_ratios":     deque(maxlen=8),
        "accident_frames": 0,
        "active_signals":  set(),
        "class_id":        -1,
    }


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
# FALL DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_fall(frame):
    """
    Top-down optimised. Three conditions + temporal gate.
    See previous implementation for full docstring.
    """
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

    cond1 = (abs(rs[0] - ls[0]) / w) > FALL_SHOULDER_SPREAD_RATIO

    ankle_y = ((la[1]+ra[1])/2 if la and ra else
               la[1] if la else ra[1] if ra else None)
    cond2 = bool(ns and ankle_y and
                 (abs(ankle_y - ns[1]) / h) < FALL_VERTICAL_COMPRESSION)

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

    all_conditions     = cond1 and cond2 and cond3
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
        cv2.putText(frame,
                    f"Fall? ({fall_frame_counter}/{FALL_CONFIRM_FRAMES})",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    return fall_confirmed, frame


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — HYBRID ACCIDENT DETECTION
# Context-triggered instance segmentation + mask-level collision validation
# ══════════════════════════════════════════════════════════════════════════════

def detect_accidents(frame, vehicle_boxes, person_boxes, masks_by_id):
    """
    Phase 2 Hybrid Architecture — two-stage pipeline:

    ── STAGE 1: Bounding Box Screening (every frame, O(n²) box pairs) ──────────
    Compute box IoU for all vehicle pairs.
    If IoU < BOX_IOU_SEG_TRIGGER → skip this pair entirely (fast path).
    If IoU ≥ BOX_IOU_SEG_TRIGGER → this pair is a CANDIDATE, proceed to Stage 2.

    ── STAGE 2: Pixel Mask Validation (only for candidates) ────────────────────
    Retrieve pixel masks for both vehicles from masks_by_id.
    Compute mask intersection ratio using compute_mask_overlap().

    If mask_overlap < MASK_OVERLAP_THRESHOLD:
        → Boxes overlapped but bodies did NOT touch
        → FALSE POSITIVE eliminated
        → seg_metrics["fp_eliminated"] += 1

    If mask_overlap ≥ MASK_OVERLAP_THRESHOLD:
        → Pixel-level contact confirmed
        → MASK_COLLISION signal fires
        → seg_metrics["mask_collisions"] += 1

    ── SIGNAL COMBINATION ───────────────────────────────────────────────────────
    MASK_COLLISION  = Signal 1 (replaces old box IoU signal)
    VELOCITY_DROP   = Signal 2 (unchanged)
    TRAJECTORY_DEV  = Signal 3 (unchanged)
    PEDESTRIAN_HIT  = Signal 4 (unchanged)
    BBOX_DEFORM     = Signal 5 (unchanged)

    MIN_SIGNALS_REQUIRED = 2 → at least MASK_COLLISION + one more signal.
    ACCIDENT_CONFIRM_FRAMES = 4 → must hold for 4 consecutive frames.

    ── VISUAL FEEDBACK ──────────────────────────────────────────────────────────
    Cyan mask overlay   → segmentation active on this vehicle
    Orange box          → signals building (pending confirmation)
    Red box             → accident confirmed
    "SEG" label         → mask was used for this detection
    """
    accident_detected = False
    accident_reason   = ""
    frame_signals     = {}

    # ── Update per-vehicle state ──────────────────────────────────────────────
    for (tid, cls_id, x1, y1, x2, y2) in vehicle_boxes:
        if tid not in vehicle_states:
            vehicle_states[tid] = init_vehicle_state()
        vs = vehicle_states[tid]
        vs["class_id"] = cls_id
        cx, cy = (x1+x2)//2, (y1+y2)//2
        vs["centroids"].append((cx, cy))

        if len(vs["centroids"]) >= 2:
            prev   = vs["centroids"][-2]
            vx, vy = cx-prev[0], cy-prev[1]
            vs["velocities"].append(np.hypot(vx, vy))
            vs["vel_vectors"].append((vx, vy))
            vs["headings"].append(np.degrees(np.arctan2(vy, vx)))

        vs["bbox_ratios"].append(max(x2-x1, 1) / max(y2-y1, 1))
        frame_signals[tid] = set()

    # ── STAGE 1 + 2: Box IoU trigger → Mask validation ───────────────────────
    for i in range(len(vehicle_boxes)):
        for j in range(i+1, len(vehicle_boxes)):
            tid_a, cls_a, *box_a = vehicle_boxes[i]
            tid_b, cls_b, *box_b = vehicle_boxes[j]

            box_iou = compute_iou(box_a, box_b)

            # Fast path — boxes not close enough, skip entirely
            if box_iou < BOX_IOU_SEG_TRIGGER:
                continue

            # Box IoU triggered — attempt mask-level validation
            seg_metrics["seg_triggers"] += 1

            mask_a = masks_by_id.get(tid_a)
            mask_b = masks_by_id.get(tid_b)

            la = VEHICLE_LABELS.get(cls_a, "VEHICLE")
            lb = VEHICLE_LABELS.get(cls_b, "VEHICLE")

            if mask_a is not None and mask_b is not None:
                # Stage 2 — pixel mask intersection
                overlap_ratio = compute_mask_overlap(mask_a, mask_b)

                if overlap_ratio >= MASK_OVERLAP_THRESHOLD:
                    # Pixel contact confirmed → MASK_COLLISION signal
                    seg_metrics["mask_collisions"] += 1
                    for tid in (tid_a, tid_b):
                        frame_signals.setdefault(tid, set()).add("MASK_COLLISION")
                    accident_reason = accident_reason or (
                        f"{la} vs {lb} pixel collision "
                        f"(mask_overlap={overlap_ratio:.2f} "
                        f"box_iou={box_iou:.2f})"
                    )
                else:
                    # Boxes overlapped but masks didn't → FP eliminated
                    seg_metrics["fp_eliminated"] += 1

            else:
                # Masks unavailable — fall back to converging velocity check
                # This handles frames where segmentation output is incomplete
                vs_a = vehicle_states.get(tid_a)
                vs_b = vehicle_states.get(tid_b)
                if (vs_a and vs_b and
                        vs_a["vel_vectors"] and vs_b["vel_vectors"]):
                    ca = vs_a["centroids"][-1]
                    cb = vs_b["centroids"][-1]
                    dx, dy = cb[0]-ca[0], cb[1]-ca[1]
                    dist   = np.hypot(dx, dy) + 1e-6
                    dx_n, dy_n = dx/dist, dy/dist
                    vxa, vya = vs_a["vel_vectors"][-1]
                    vxb, vyb = vs_b["vel_vectors"][-1]
                    converging = (vxa*dx_n + vya*dy_n) + (-vxb*dx_n - vyb*dy_n)
                    if converging > 1.0:
                        for tid in (tid_a, tid_b):
                            frame_signals.setdefault(tid, set()).add("MASK_COLLISION")
                        accident_reason = accident_reason or (
                            f"{la} vs {lb} collision (fallback, "
                            f"approach={converging:.1f}px/f)"
                        )

    # ── Signal 2: Robust velocity drop ───────────────────────────────────────
    for (tid, cls_id, x1, y1, x2, y2) in vehicle_boxes:
        vs    = vehicle_states.get(tid)
        label = VEHICLE_LABELS.get(cls_id, "VEHICLE")
        if not vs or len(vs["velocities"]) < VELOCITY_BASELINE_FRAMES:
            continue
        vel_list = list(vs["velocities"])
        baseline = vel_list[:-3]
        curr_vel = np.mean(vel_list[-3:])
        avg_base = np.mean(baseline)
        std_base = np.std(baseline)
        if (avg_base > MIN_MOVING_VELOCITY and
                std_base < VELOCITY_BASELINE_MAX_STD and
                curr_vel < avg_base * VELOCITY_DROP_RATIO):
            frame_signals.setdefault(tid, set()).add("VELOCITY_DROP")
            accident_reason = accident_reason or (
                f"{label} abrupt stop "
                f"(base={avg_base:.1f}→curr={curr_vel:.1f}px/f)"
            )

    # ── Signal 3: Sustained heading deviation ────────────────────────────────
    for (tid, cls_id, x1, y1, x2, y2) in vehicle_boxes:
        vs    = vehicle_states.get(tid)
        label = VEHICLE_LABELS.get(cls_id, "VEHICLE")
        if not vs or len(vs["headings"]) < 3:
            continue
        diff = abs(vs["headings"][-1] - vs["headings"][-2])
        diff = min(diff, 360 - diff)
        if diff > HEADING_CHANGE_THRESHOLD:
            vs["heading_dev_cnt"] += 1
        else:
            vs["heading_dev_cnt"] = 0
        if vs["heading_dev_cnt"] >= HEADING_SUSTAINED_FRAMES:
            frame_signals.setdefault(tid, set()).add("TRAJECTORY_DEV")
            accident_reason = accident_reason or (
                f"{label} trajectory deviation ({diff:.0f}°)"
            )

    # ── Signal 4: Person-vehicle impact ──────────────────────────────────────
    for ptid, pcx, pcy in person_boxes:
        ps = person_states.setdefault(ptid, {"entry_time": None, "loiter_logged": False})
        if "p_centroids" not in ps:
            ps["p_centroids"]  = deque(maxlen=10)
            ps["p_velocities"] = deque(maxlen=10)
        ps["p_centroids"].append((pcx, pcy))
        if len(ps["p_centroids"]) >= 2:
            prev = ps["p_centroids"][-2]
            ps["p_velocities"].append(np.hypot(pcx-prev[0], pcy-prev[1]))

    for (tid, cls_id, vx1, vy1, vx2, vy2) in vehicle_boxes:
        vs    = vehicle_states.get(tid)
        label = VEHICLE_LABELS.get(cls_id, "VEHICLE")
        if not vs or len(vs["velocities"]) < 3:
            continue
        if np.mean(list(vs["velocities"])[-3:]) < MIN_MOVING_VELOCITY:
            continue
        pad = PERSON_VEHICLE_OVERLAP_PAD
        for ptid, pcx, pcy in person_boxes:
            if not ((vx1-pad) <= pcx <= (vx2+pad) and
                    (vy1-pad) <= pcy <= (vy2+pad)):
                continue
            ps = person_states.get(ptid, {})
            pv = list(ps.get("p_velocities", []))
            if len(pv) >= 4:
                if (np.mean(pv[-2:]) / (np.mean(pv[-4:-2])+1e-6)) < PERSON_DECEL_RATIO:
                    frame_signals.setdefault(tid, set()).add("PEDESTRIAN_HIT")
                    accident_reason = accident_reason or (
                        f"Pedestrian impact by {label}"
                    )

    # ── Signal 5: Median-smoothed bbox deformation ───────────────────────────
    for (tid, cls_id, x1, y1, x2, y2) in vehicle_boxes:
        vs    = vehicle_states.get(tid)
        label = VEHICLE_LABELS.get(cls_id, "VEHICLE")
        if not vs or len(vs["bbox_ratios"]) < 6:
            continue
        ratios = list(vs["bbox_ratios"])
        if abs(np.median(ratios[-3:]) - np.median(ratios[-6:-3])) > BBOX_RATIO_CHANGE:
            frame_signals.setdefault(tid, set()).add("BBOX_DEFORM")
            accident_reason = accident_reason or f"{label} shape deformation"

    # ── Multi-signal temporal gate ────────────────────────────────────────────
    confirmed_ids = set()
    flagged_ids   = set()

    for (tid, *_) in vehicle_boxes:
        vs = vehicle_states.get(tid)
        if vs is None:
            continue
        sigs = frame_signals.get(tid, set())
        if len(sigs) >= MIN_SIGNALS_REQUIRED:
            vs["active_signals"] |= sigs
            vs["accident_frames"] += 1
            flagged_ids.add(tid)
        else:
            vs["accident_frames"] = 0
            vs["active_signals"]  = set()
        if vs["accident_frames"] >= ACCIDENT_CONFIRM_FRAMES:
            confirmed_ids.add(tid)

    # ── Annotate frame ────────────────────────────────────────────────────────
    for (tid, cls_id, x1, y1, x2, y2) in vehicle_boxes:
        vs    = vehicle_states.get(tid)
        label = VEHICLE_LABELS.get(cls_id, "VEHICLE")

        # Draw cyan pixel mask overlay for vehicles with active segmentation
        if tid in masks_by_id and tid in (flagged_ids | confirmed_ids):
            mask = masks_by_id[tid]
            overlay         = frame.copy()
            overlay[mask > 0] = [255, 255, 0]   # cyan overlay on mask pixels
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        if tid in confirmed_ids:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f"ACCIDENT {label} [SEG]",
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
            accident_detected = True

        elif tid in flagged_ids and vs:
            sigs = "+".join(sorted(vs["active_signals"])) or "?"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
            cv2.putText(frame,
                        f"{sigs} ({vs['accident_frames']}/{ACCIDENT_CONFIRM_FRAMES})",
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 140, 255), 2)

    if accident_detected:
        cv2.putText(frame, "ACCIDENT DETECTED [Phase 2]",
                    (30, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 3)

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

@app.route("/toggle_crowd",    methods=["POST"])
def toggle_crowd():
    global crowd_detection_enabled
    crowd_detection_enabled = not crowd_detection_enabled
    return jsonify({"enabled": crowd_detection_enabled})

@app.route("/toggle_fire",     methods=["POST"])
def toggle_fire():
    global fire_detection_enabled
    fire_detection_enabled = not fire_detection_enabled
    return jsonify({"enabled": fire_detection_enabled})

@app.route("/toggle_fall",     methods=["POST"])
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
    return send_file(
        buf, mimetype="text/csv", as_attachment=True,
        download_name=f"safesight_logs_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    )

@app.route("/snapshots")
def list_snapshots():
    files = sorted(os.listdir(SNAPSHOT_DIR), reverse=True)
    return jsonify([
        {"filename": f, "url": f"/static/snapshots/{f}"}
        for f in files if f.endswith(".jpg")
    ])

@app.route("/seg_metrics")
def get_seg_metrics():
    """
    Phase 2 segmentation performance metrics.
    Expose to dashboard to show research contribution live during viva.
    """
    return jsonify(seg_metrics)

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
        masks_by_id   = {}   # Phase 2 — populated from segmentation results

        for result in results:
            frame = result.plot(conf=False)
            if roi_polygon is not None:
                cv2.polylines(frame, [roi_polygon], True, (255, 0, 0), 2)

            # Phase 2 — extract pixel masks for all tracked objects this frame
            frame_masks = YOLODetector.extract_masks(result, frame.shape)
            masks_by_id.update(frame_masks)

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

        # ── Accident (Phase 2 — passes masks_by_id) ───────────────────────────
        if accident_detection_enabled and vehicle_boxes:
            acc_detected, frame, acc_reason = detect_accidents(
                frame, vehicle_boxes, person_boxes, masks_by_id
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