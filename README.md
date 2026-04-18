# 🤖 Autonomous Rover Vision Navigation Assistant

Real-time obstacle detection and navigation decision-making using a laptop webcam, YOLOv8, and OpenCV.

---

## ⚡ Quick Start

```bash
pip install -r requirements.txt
python rover_nav.py
```

---

## 🗺 How It Works

The camera frame is divided into **three vertical zones**:

```
┌───────────┬───────────┬───────────┐
│           │           │           │
│   LEFT    │  CENTER   │   RIGHT   │
│           │           │           │
└───────────┴───────────┴───────────┘
```

Each detected object is assigned to the zone containing its bounding-box centre.  
A zone is marked **BLOCKED** when obstacle coverage exceeds 8% of that zone's area.

### Navigation decisions

| Zone state                         | Decision                  |
|------------------------------------|---------------------------|
| CENTER clear                       | MOVE FORWARD              |
| CENTER blocked, LEFT clear         | TURN LEFT                 |
| CENTER blocked, RIGHT clear        | TURN RIGHT                |
| All zones blocked                  | STOP                      |
| LEFT + RIGHT blocked, CENTER clear | FORWARD WITH CAUTION      |
| Person detected in CENTER          | HUMAN DETECTED — STOP ⚠   |

---

## 🎮 Controls

| Key | Action                        |
|-----|-------------------------------|
| Q   | Quit                          |
| S   | Save screenshot               |
| +   | Raise confidence threshold    |
| -   | Lower confidence threshold    |

---

## 🛠 Configuration (top of rover_nav.py)

| Variable              | Default      | Effect                                    |
|-----------------------|-------------|-------------------------------------------|
| `CONFIDENCE_THRESHOLD`| `0.45`      | Min detection score shown                 |
| `MODEL_NAME`          | `yolov8n.pt`| Model size — n/s/m/l                      |
| `WEBCAM_INDEX`        | `0`         | 0 = built-in, 1 = USB cam                |
| `ZONE_BLOCK_RATIO`    | `0.08`      | Coverage fraction to consider zone blocked|
| `SOUND_ALERT_ENABLED` | `True`      | Terminal bell on STOP events              |

---

## 📁 Project Structure

```
rover_vision/
├── rover_nav.py       ← main script
├── requirements.txt
├── README.md
└── screenshots/       ← auto-created when you press S
```

---

## 🔗 Relation to Project 1

| Feature                        | Project 1 | Project 2 (this) |
|-------------------------------|-----------|------------------|
| Webcam capture                | ✅        | ✅ (reused)       |
| YOLO detection + boxes        | ✅        | ✅ (reused)       |
| FPS counter                   | ✅        | ✅ (reused)       |
| Confidence threshold controls | ✅        | ✅ (reused)       |
| Screenshot (S key)            | ✅        | ✅ (reused)       |
| Zone division (L/C/R)         | —         | ✅                |
| Navigation logic engine       | —         | ✅                |
| Human safety override         | —         | ✅                |
| Nearest obstacle label        | —         | ✅                |
| Obstacle priority by size     | —         | ✅                |
| Sound alert                   | —         | ✅                |

---

Built with ❤️ for college rover team demonstration.  
Uses [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and [OpenCV](https://opencv.org/).
