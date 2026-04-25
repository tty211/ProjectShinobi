import os
import asyncio
import base64
import json
import cv2
import torch
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
import uvicorn

HAND_MODEL_PATH = "best.pt"
GESTURE_MODEL_PATH = "best_gesture.pt"
HAND_CONF = 0.35
HAND_IOU = 0.45
CROP_PADDING = 20

app = FastAPI()
app.mount("/images", StaticFiles(directory="images"), name="images")
app.mount("/background", StaticFiles(directory="background"), name="background")

@app.get("/")
def index():
    return FileResponse("index.html")

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

def load_models():
    from ultralytics import YOLO
    device = 0 if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"[GPU] 使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[CPU] 未检测到 CUDA，使用 CPU 推理（速度较慢）")
    hand_model = YOLO(HAND_MODEL_PATH).to("cuda" if device == 0 else "cpu")
    gesture_model = YOLO(GESTURE_MODEL_PATH).to("cuda" if device == 0 else "cpu")
    return hand_model, gesture_model, device

def clamp(v, lo, hi):
    return max(lo, min(v, hi))

def process_frame(frame, hand_model, gesture_model, device):
    h, w = frame.shape[:2]
    gesture_name = ""
    gesture_conf = 0.0

    hand_results = hand_model.predict(source=frame, conf=HAND_CONF, iou=HAND_IOU, verbose=False, device=device)
    hand_result = hand_results[0]

    if hand_result.boxes is not None and len(hand_result.boxes) > 0:
        confs = hand_result.boxes.conf.cpu().numpy()
        best_idx = int(confs.argmax())
        x1, y1, x2, y2 = map(int, hand_result.boxes.xyxy.cpu().numpy()[best_idx])

        x1p = clamp(x1 - CROP_PADDING, 0, w - 1)
        y1p = clamp(y1 - CROP_PADDING, 0, h - 1)
        x2p = clamp(x2 + CROP_PADDING, 0, w - 1)
        y2p = clamp(y2 + CROP_PADDING, 0, h - 1)

        hand_crop = frame[y1p:y2p, x1p:x2p]
        cv2.rectangle(frame, (x1p, y1p), (x2p, y2p), (0, 255, 0), 2)

        if hand_crop.size > 0:
            g_results = gesture_model.predict(source=hand_crop, conf=0.0, iou=0.0, verbose=False, device=device)
            g_result = g_results[0]
            if g_result.boxes is not None and len(g_result.boxes) > 0:
                g_confs = g_result.boxes.conf.cpu().numpy()
                best_g = int(g_confs.argmax())
                gesture_conf = float(g_confs[best_g])
                gesture_name = g_result.names[int(g_result.boxes.cls.cpu().numpy()[best_g])]

    return frame, gesture_name, gesture_conf

def find_camera():
    for i in range(6):
        c = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if c.isOpened():
            ok, f = c.read()
            if ok and f is not None:
                # 降低分辨率减少推理耗时
                c.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                return c
            c.release()
    return None

def inference_loop(hand_model, gesture_model, device, cap, result_holder):
    """在独立线程中持续推理，结果写入 result_holder"""
    import time
    min_interval = 0.08  # 最多 ~12fps 推理，CPU 下够用
    last_t = 0
    while result_holder["running"]:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # 水平镜像，修正摄像头左右翻转
        now = time.time()
        if now - last_t < min_interval:
            time.sleep(0.01)
            continue
        last_t = now
        frame, gesture_name, gesture_conf = process_frame(frame, hand_model, gesture_model, device)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 55])
        result_holder["data"] = {
            "gesture": gesture_name,
            "confidence": round(gesture_conf, 3),
            "frame": base64.b64encode(buf).decode()
        }
    cap.release()

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    hand_model, gesture_model, device = load_models()

    cap = find_camera()
    if cap is None:
        await ws.send_text(json.dumps({"error": "No camera found"}))
        return

    result_holder = {"running": True, "data": None}
    loop = asyncio.get_event_loop()
    thread = loop.run_in_executor(None, inference_loop, hand_model, gesture_model, device, cap, result_holder)

    try:
        last_sent = None
        send_interval = 0.08  # 与推理帧率匹配，约 12fps
        last_send_t = 0
        while True:
            data = result_holder["data"]
            now = asyncio.get_event_loop().time()
            if data is not None and data is not last_sent and now - last_send_t >= send_interval:
                last_sent = data
                last_send_t = now
                await ws.send_text(json.dumps(data))
            else:
                await asyncio.sleep(0.01)
    except WebSocketDisconnect:
        pass
    finally:
        result_holder["running"] = False
        await thread
        os._exit(0)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
