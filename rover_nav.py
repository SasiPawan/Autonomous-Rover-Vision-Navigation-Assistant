import cv2
import time
import os
from datetime import datetime
from ultralytics import YOLO

CONFIDENCE_THRESHOLD = 0.45
MODEL_NAME = "yolov8n.pt"
WEBCAM_INDEX = 0
SCREENSHOT_DIR = "screenshots"

ZONE_BLOCK_RATIO = 0.08
ZONE_MIN_AREA = 6000
STABLE_FRAMES = 3

HIGH_PRIORITY_CLASSES = {
    "person", "bicycle", "car", "motorcycle",
    "bus", "truck", "dog", "cat", "horse", "cow"
}

PALETTE = [
    (56, 210, 255), (255, 86, 56), (56, 255, 130), (255, 200, 56),
    (180, 56, 255), (56, 140, 255), (255, 56, 180), (56, 255, 220),
    (255, 140, 56), (130, 255, 56)
]

NAV_COLORS = {
    "MOVE FORWARD": (80, 230, 80),
    "TURN LEFT": (255, 200, 50),
    "TURN RIGHT": (255, 200, 50),
    "STOP": (50, 50, 255),
    "FORWARD WITH CAUTION": (50, 190, 255),
    "HUMAN DETECTED - STOP": (30, 30, 255),
}

def get_color(class_id):
    return PALETTE[class_id % len(PALETTE)]

def draw_detection_box(frame, x1, y1, x2, y2, color, label):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thick = 1

    (tw, th), bl = cv2.getTextSize(label, font, scale, thick)

    by1 = y1 - th - bl - 6
    by2 = y1

    if by1 < 0:
        by1, by2 = y1, y1 + th + bl + 6

    cv2.rectangle(frame, (x1, by1), (x1 + tw + 8, by2), color, cv2.FILLED)

    cv2.putText(frame, label, (x1 + 4, by2 - bl - 2),
                font, scale, (0, 0, 0), thick, cv2.LINE_AA)

def draw_zones(frame, zones_blocked, zone_x):
    h, w = frame.shape[:2]
    lx, rx = zone_x

    zone_rects = {
        "LEFT": (0, 0, lx, h),
        "CENTER": (lx, 0, rx, h),
        "RIGHT": (rx, 0, w, h)
    }

    overlay = frame.copy()

    for name, (ax, ay, bx, by) in zone_rects.items():
        if zones_blocked[name]:
            cv2.rectangle(overlay, (ax, ay), (bx, by), (30, 30, 200), cv2.FILLED)

    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

    cv2.line(frame, (lx, 0), (lx, h), (200, 200, 200), 1)
    cv2.line(frame, (rx, 0), (rx, h), (200, 200, 200), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    label_y = h - 12

    for name, (ax, ay, bx, by) in zone_rects.items():
        cx = (ax + bx) // 2
        (tw, _), _ = cv2.getTextSize(name, font, 0.45, 1)

        color = (50, 50, 255) if zones_blocked[name] else (180, 180, 180)

        cv2.putText(frame, name, (cx - tw // 2, label_y),
                    font, 0.45, color, 1, cv2.LINE_AA)

def draw_hud(frame, fps, obj_count, decision, nearest_label, safety_alert):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 75), (10, 10, 10), cv2.FILLED)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    font_s = cv2.FONT_HERSHEY_SIMPLEX
    font_b = cv2.FONT_HERSHEY_DUPLEX

    fps_col = (80, 255, 80) if fps >= 20 else (0, 220, 255) if fps >= 10 else (50, 50, 255)

    cv2.putText(frame, f"FPS {fps:4.1f}", (12, 22), font_b, 0.55, fps_col, 1, cv2.LINE_AA)
    cv2.putText(frame, f"OBJ {obj_count}", (120, 22), font_b, 0.55, (255, 215, 50), 1, cv2.LINE_AA)

    hint = "Q:Quit   S:Screenshot"
    (tw, _), _ = cv2.getTextSize(hint, font_s, 0.44, 1)

    cv2.putText(frame, hint, (w - tw - 10, 22),
                font_s, 0.44, (130, 130, 130), 1, cv2.LINE_AA)

    dec_col = NAV_COLORS.get(decision, (255, 255, 255))

    cv2.putText(frame, decision, (12, 55),
                font_b, 0.72, dec_col, 1, cv2.LINE_AA)

    if nearest_label:
        txt = f"Nearest: {nearest_label}"
        (tw, _), _ = cv2.getTextSize(txt, font_s, 0.44, 1)

        cv2.putText(frame, txt, (w - tw - 10, 55),
                    font_s, 0.44, (255, 190, 60), 1, cv2.LINE_AA)

    if safety_alert:
        cv2.putText(frame, "ALERT", (w - 80, 70),
                    font_b, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

def classify_detections(results, model_names, frame_w, frame_h, zone_x):
    lx, rx = zone_x

    zone_areas = {"LEFT": 0, "CENTER": 0, "RIGHT": 0}
    zone_widths = {
        "LEFT": lx,
        "CENTER": rx - lx,
        "RIGHT": frame_w - rx
    }

    detections = []
    human_in_center = False
    largest_area = 0
    nearest_label = ""

    for box in results.boxes:
        conf = float(box.conf[0])

        if conf < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)

        if area < ZONE_MIN_AREA:
            continue

        cid = int(box.cls[0])
        name = model_names[cid]

        cx = (x1 + x2) // 2

        if cx < lx:
            zone = "LEFT"
        elif cx < rx:
            zone = "CENTER"
        else:
            zone = "RIGHT"

        zone_areas[zone] += area

        if area > largest_area:
            largest_area = area
            nearest_label = f"{name} ({conf:.0%})"

        if name == "person" and zone == "CENTER":
            human_in_center = True

        detections.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "class_id": cid,
            "name": name,
            "conf": conf,
            "area": area
        })

    zones_blocked = {}

    for zn, zw in zone_widths.items():
        zone_area_total = zw * frame_h
        ratio = zone_areas[zn] / max(zone_area_total, 1)
        zones_blocked[zn] = ratio >= ZONE_BLOCK_RATIO

    return detections, zones_blocked, human_in_center, nearest_label

def decide_navigation(zones_blocked, human_in_center):
    L = zones_blocked["LEFT"]
    C = zones_blocked["CENTER"]
    R = zones_blocked["RIGHT"]

    if human_in_center:
        return "HUMAN DETECTED - STOP", True

    if not C:
        if L and R:
            return "FORWARD WITH CAUTION", False
        return "MOVE FORWARD", False

    if not L:
        return "TURN LEFT", False

    if not R:
        return "TURN RIGHT", False

    return "STOP", True

def main():

    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(WEBCAM_INDEX)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    prev_time = time.time()
    fps_history = []

    current_decision = "MOVE FORWARD"
    pending_decision = "MOVE FORWARD"
    decision_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        h, w = frame.shape[:2]

        lx = w // 3
        rx = (2 * w) // 3
        zone_x = (lx, rx)

        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

        detections, zones_blocked, human_in_center, nearest_label = classify_detections(
            results, model.names, w, h, zone_x
        )

        raw_decision, safety_alert = decide_navigation(
            zones_blocked, human_in_center
        )

        if raw_decision == pending_decision:
            decision_count += 1
        else:
            pending_decision = raw_decision
            decision_count = 1

        if decision_count >= STABLE_FRAMES:
            current_decision = pending_decision

        for det in sorted(detections, key=lambda d: d["area"], reverse=True):
            color = get_color(det["class_id"])

            if det["name"] in HIGH_PRIORITY_CLASSES:
                color = tuple(min(c + 60, 255) for c in color)

            label = f"{det['name']} {det['conf']:.0%}"

            draw_detection_box(
                frame,
                det["x1"], det["y1"],
                det["x2"], det["y2"],
                color, label
            )

        curr_time = time.time()
        fps_history.append(1.0 / max(curr_time - prev_time, 1e-6))
        prev_time = curr_time

        if len(fps_history) > 15:
            fps_history.pop(0)

        fps = sum(fps_history) / len(fps_history)

        draw_zones(frame, zones_blocked, zone_x)
        draw_hud(frame, fps, len(detections), current_decision, nearest_label, safety_alert)

        cv2.imshow("Autonomous Rover Vision Navigation", frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q')):
            break

        elif key in (ord('s'), ord('S')):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SCREENSHOT_DIR, f"rover_{ts}.jpg")
            cv2.imwrite(path, frame)
            print("Screenshot saved:", path)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
