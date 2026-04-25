import cv2
import torch
from ultralytics import YOLO

HAND_MODEL_PATH = "best.pt"
GESTURE_MODEL_PATH = "best_gesture.pt"

HAND_CONF = 0.35
HAND_IOU = 0.45

GESTURE_CONF = 0.0
GESTURE_IOU = 0.0

CROP_PADDING = 20
SHOW_ALL_HANDS = False
MAX_CAMERAS_TO_CHECK = 6


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def pick_best_box(result):
    if result.boxes is None or len(result.boxes) == 0:
        return None

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    best_idx = int(confs.argmax())
    x1, y1, x2, y2 = boxes[best_idx]
    conf = float(confs[best_idx])
    cls_id = int(classes[best_idx])

    return {
        "xyxy": (int(x1), int(y1), int(x2), int(y2)),
        "conf": conf,
        "cls_id": cls_id,
    }


def draw_label(img, text, x, y, color=(0, 255, 0)):
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA
    )


def find_available_cameras(max_index=6):
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                available.append(i)
            cap.release()
    return available


def choose_camera(max_index=6):
    cameras = find_available_cameras(max_index)
    if not cameras:
        raise RuntimeError("No webcam found.")

    print("Available camera indexes:", cameras)
    print("Opening preview windows...")

    preview_caps = {}
    for idx in cameras:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            preview_caps[idx] = cap

    selected = None

    while True:
        for idx, cap in list(preview_caps.items()):
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            preview = frame.copy()
            cv2.putText(
                preview,
                f"Camera {idx} - press {idx} to select",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )
            cv2.imshow(f"Camera {idx}", preview)

        key = cv2.waitKey(1) & 0xFF

        if ord("0") <= key <= ord("9"):
            chosen = key - ord("0")
            if chosen in cameras:
                selected = chosen
                break

        if key == 27 or key == ord("q"):
            break

    for idx, cap in preview_caps.items():
        cap.release()
        cv2.destroyWindow(f"Camera {idx}")

    if selected is None:
        raise RuntimeError("No camera selected.")

    print(f"Selected camera: {selected}")
    return selected


def main():
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    print("Using device:", DEVICE)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    camera_index = choose_camera(MAX_CAMERAS_TO_CHECK)

    hand_model = YOLO(HAND_MODEL_PATH)
    gesture_model = YOLO(GESTURE_MODEL_PATH)

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam {camera_index}.")

    cv2.namedWindow("Hand + Gesture Test", cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from webcam.")
            break

        display = frame.copy()
        h, w = frame.shape[:2]

        hand_results = hand_model.predict(
            source=frame,
            conf=HAND_CONF,
            iou=HAND_IOU,
            verbose=False,
            device=DEVICE
        )

        hand_result = hand_results[0]

        if SHOW_ALL_HANDS and hand_result.boxes is not None and len(hand_result.boxes) > 0:
            for box, conf, cls_id in zip(
                hand_result.boxes.xyxy.cpu().numpy(),
                hand_result.boxes.conf.cpu().numpy(),
                hand_result.boxes.cls.cpu().numpy()
            ):
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 180, 0), 1)
                label = hand_result.names[int(cls_id)]
                draw_label(display, f"{label} {conf:.2f}", x1, max(20, y1 - 8), (255, 180, 0))

        best_hand = pick_best_box(hand_result)

        if best_hand is not None:
            x1, y1, x2, y2 = best_hand["xyxy"]

            x1p = clamp(x1 - CROP_PADDING, 0, w - 1)
            y1p = clamp(y1 - CROP_PADDING, 0, h - 1)
            x2p = clamp(x2 + CROP_PADDING, 0, w - 1)
            y2p = clamp(y2 + CROP_PADDING, 0, h - 1)

            hand_crop = frame[y1p:y2p, x1p:x2p]

            cv2.rectangle(display, (x1p, y1p), (x2p, y2p), (0, 255, 0), 2)
            draw_label(
                display,
                f"Hand {best_hand['conf']:.2f}",
                x1p,
                max(20, y1p - 10),
                (0, 255, 0)
            )

            if hand_crop.size > 0:
                gesture_results = gesture_model.predict(
                    source=hand_crop,
                    conf=GESTURE_CONF,
                    iou=GESTURE_IOU,
                    verbose=False,
                    device=DEVICE
                )

                gesture_result = gesture_results[0]

                if gesture_result.boxes is not None and len(gesture_result.boxes) > 0:
                    g_boxes = gesture_result.boxes.xyxy.cpu().numpy()
                    g_confs = gesture_result.boxes.conf.cpu().numpy()
                    g_classes = gesture_result.boxes.cls.cpu().numpy()

                    best_g_idx = int(g_confs.argmax())
                    gx1, gy1, gx2, gy2 = map(int, g_boxes[best_g_idx][:4])
                    g_conf = float(g_confs[best_g_idx])
                    g_cls = int(g_classes[best_g_idx])
                    g_name = gesture_result.names[g_cls]

                    cv2.rectangle(display, (x1p + gx1, y1p + gy1), (x1p + gx2, y1p + gy2), (0, 0, 255), 2)
                    draw_label(
                        display,
                        f"{g_name} {g_conf:.2f}",
                        x1p + gx1,
                        max(20, y1p + gy1 - 10),
                        (0, 0, 255)
                    )

                    cv2.putText(
                        display,
                        f"Gesture: {g_name}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA
                    )
                else:
                    cv2.putText(
                        display,
                        "Gesture: none",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (100, 100, 255),
                        2,
                        cv2.LINE_AA
                    )

                preview = cv2.resize(hand_crop, (220, 220))
                ph, pw = preview.shape[:2]

                if 10 + ph <= h and w - pw - 10 >= 0:
                    display[10:10 + ph, w - pw - 10:w - 10] = preview
                    cv2.rectangle(display, (w - pw - 10, 10), (w - 10, 10 + ph), (255, 255, 255), 2)

        else:
            cv2.putText(
                display,
                "No hand detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

        cv2.imshow("Hand + Gesture Test", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()