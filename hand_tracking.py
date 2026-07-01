import argparse
import math
import random
import time
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)
INDEX_FINGER_TIP = 8
WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17
FINGER_TIPS = (8, 12, 16, 20)
FINGER_PIPS = (6, 10, 14, 18)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Track hand landmarks from a webcam with OpenCV and MediaPipe."
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument("--width", type=int, default=1280, help="Capture width.")
    parser.add_argument("--height", type=int, default=720, help="Capture height.")
    parser.add_argument("--max-hands", type=int, default=2, help="Maximum hands to track.")
    parser.add_argument(
        "--detection-confidence",
        type=float,
        default=0.6,
        help="Minimum detection confidence, from 0.0 to 1.0.",
    )
    parser.add_argument(
        "--tracking-confidence",
        type=float,
        default=0.6,
        help="Minimum tracking confidence, from 0.0 to 1.0.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the MediaPipe hand landmarker .task model.",
    )
    parser.add_argument(
        "--heart-hand",
        choices=("left", "right", "both"),
        default="left",
        help="Which hand should create hearts when opened toward the camera.",
    )
    parser.add_argument("--show-fps", action="store_true", help="Show FPS counter.")
    return parser.parse_args()


def ensure_model(model_path):
    if model_path.exists():
        return

    print("Downloading MediaPipe hand model...")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urlretrieve(MODEL_URL, model_path)
    except Exception as exc:
        raise RuntimeError(
            "The hand model could not be downloaded. Check your internet connection "
            f"or manually download it from:\n{MODEL_URL}\n"
            f"Then save it here:\n{model_path}"
        ) from exc


def put_text(image, text, x, y, color=(20, 240, 20)):
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def to_pixel(landmark, width, height):
    x = int(landmark.x * width)
    y = int(landmark.y * height)
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    return x, y


def category_name(category):
    name = getattr(category, "category_name", "")
    if name:
        return name
    label = getattr(category, "label", "")
    if label:
        return label
    return "Hand"


def corrected_label(label):
    if label == "Left":
        return "Right"
    if label == "Right":
        return "Left"
    return label


def distance_2d(first, second):
    return math.hypot(first.x - second.x, first.y - second.y)


def is_open_palm(landmarks):
    wrist = landmarks[WRIST]
    extended_fingers = 0

    for tip_id, pip_id in zip(FINGER_TIPS, FINGER_PIPS):
        tip = landmarks[tip_id]
        pip = landmarks[pip_id]

        farther_from_wrist = distance_2d(wrist, tip) > distance_2d(wrist, pip) * 1.05
        higher_than_joint = tip.y < pip.y + 0.03

        if farther_from_wrist and higher_than_joint:
            extended_fingers += 1

    palm_width = distance_2d(landmarks[INDEX_MCP], landmarks[PINKY_MCP])
    palm_height = distance_2d(landmarks[WRIST], landmarks[MIDDLE_MCP])
    palm_is_visible_size = palm_width > palm_height * 0.35

    return extended_fingers >= 3 and palm_is_visible_size


def hand_matches_heart_setting(label, setting):
    label = label.lower()
    return setting == "both" or setting == label


def draw_heart(image, center, size, color):
    points = []
    scale = size / 34.0

    for step in range(36):
        t = 2 * math.pi * step / 36
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        points.append((int(center[0] + x * scale), int(center[1] - y * scale)))

    polygon = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(image, [polygon], color, lineType=cv2.LINE_AA)
    cv2.polylines(image, [polygon], True, (255, 255, 255), 1, cv2.LINE_AA)


class HeartEmitter:
    def __init__(self):
        self.hearts = []
        self.last_spawn = 0.0
        self.colors = (
            (255, 70, 210),  # purple-pink
            (80, 255, 120),  # green
            (255, 150, 60),  # blue in OpenCV BGR
        )

    def spawn(self, origin, now):
        self.hearts.append(
            {
                "x": origin[0] + random.uniform(-16, 16),
                "y": origin[1] + random.uniform(-10, 10),
                "vx": random.uniform(-32, 32),
                "vy": random.uniform(-120, -70),
                "size": random.uniform(34, 58),
                "born": now,
                "life": random.uniform(1.0, 1.6),
                "color": random.choice(self.colors),
            }
        )

    def update_and_draw(self, frame, origin):
        now = time.perf_counter()

        if origin and now - self.last_spawn > 0.05:
            for _ in range(2):
                self.spawn(origin, now)
            self.last_spawn = now

        alive = []
        for heart in self.hearts:
            age = now - heart["born"]
            if age > heart["life"]:
                continue

            progress = age / heart["life"]
            x = heart["x"] + heart["vx"] * age
            y = heart["y"] + heart["vy"] * age
            size = heart["size"] * (1.0 - 0.25 * progress)
            base_color = heart["color"]
            color = tuple(max(20, int(channel * (1.0 - 0.35 * progress))) for channel in base_color)

            draw_heart(frame, (int(x), int(y)), int(size), color)
            alive.append(heart)

        self.hearts = alive[-120:]


def draw_hand(frame, landmarks, handedness, heart_hand):
    height, width, _ = frame.shape
    points = [to_pixel(landmark, width, height) for landmark in landmarks]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (20, 220, 20), 3)

    for point in points:
        cv2.circle(frame, point, 5, (0, 120, 255), -1)

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    cv2.rectangle(
        frame,
        (max(0, x_min - 12), max(0, y_min - 12)),
        (min(width - 1, x_max + 12), min(height - 1, y_max + 12)),
        (20, 240, 20),
        2,
    )

    index_tip_x, index_tip_y = points[INDEX_FINGER_TIP]
    cv2.circle(frame, (index_tip_x, index_tip_y), 9, (255, 40, 40), -1)

    label = "Hand"
    score = 0.0
    if handedness:
        category = handedness[0]
        score = getattr(category, "score", 0.0)
        label = corrected_label(category_name(category))

    display_label = f"{label} {score:.2f}" if score else label
    put_text(frame, display_label, max(x_min - 12, 10), max(y_min - 20, 30))
    put_text(
        frame,
        f"Index tip: {index_tip_x}, {index_tip_y}",
        max(x_min - 12, 10),
        min(y_max + 35, height - 15),
        (0, 180, 255),
    )

    palm_center = points[MIDDLE_MCP]
    palm_visible = is_open_palm(landmarks)
    should_emit_heart = palm_visible and hand_matches_heart_setting(label, heart_hand)

    if should_emit_heart:
        cv2.circle(frame, palm_center, 13, (255, 80, 180), 2, cv2.LINE_AA)
        put_text(
            frame,
            f"{label.upper()} PALM -> HEARTS",
            max(x_min - 12, 10),
            min(y_max + 62, height - 15),
            (255, 80, 180),
        )

    return label, should_emit_heart, palm_center


def main():
    args = parse_args()
    ensure_model(args.model)

    if not hasattr(mp, "tasks"):
        raise RuntimeError(
            "This MediaPipe installation does not include the Tasks API. "
            "Run: python -m pip install --upgrade mediapipe"
        )

    base_options = mp.tasks.BaseOptions(model_asset_path=str(args.model))
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=args.max_hands,
        min_hand_detection_confidence=args.detection_confidence,
        min_hand_presence_confidence=args.detection_confidence,
        min_tracking_confidence=args.tracking_confidence,
    )

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError(
            f"Camera {args.camera} could not be opened. Try --camera 1 or check permissions."
        )

    previous_time = time.perf_counter()
    start_time = time.perf_counter()
    heart_emitter = HeartEmitter()

    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Empty camera frame; exiting.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.perf_counter() - start_time) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            heart_origin = None
            if result.hand_landmarks:
                for index, landmarks in enumerate(result.hand_landmarks):
                    handedness = []
                    if result.handedness and index < len(result.handedness):
                        handedness = result.handedness[index]
                    label, should_emit_heart, palm_center = draw_hand(
                        frame, landmarks, handedness, args.heart_hand
                    )
                    if should_emit_heart:
                        heart_origin = palm_center

            heart_emitter.update_and_draw(frame, heart_origin)

            if args.show_fps:
                now = time.perf_counter()
                fps = 1.0 / max(now - previous_time, 1e-6)
                previous_time = now
                put_text(frame, f"FPS: {fps:.1f}", 12, 30, (255, 255, 255))

            cv2.imshow("Hand Tracking - press q or ESC to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
