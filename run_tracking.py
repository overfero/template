from ultralytics import YOLO
import time

# model_name = "models/yolo26s_cpu.onnx"
# model_name = "ultralytics/checkpoint/yolo11n_openvino_model"  # native OpenVINO
model_name = "ultralytics/checkpoint/yolo11n_cpu.onnx"  # onnxruntime + OpenVINOExecutionProvider

model = YOLO(model_name)

N_FRAMES = 2268

# PENTING: stream=True agar hasil tidak akumulasi di RAM
# Tanpa stream=True → 3648 * (1920x1080x3 uint8) ≈ 22GB RAM → GC pressure → +33ms/frame
t_start = time.perf_counter()
frame_count = 0
for r in model.track(
    source="/home/overfero/Project/glair/Jumpstart - Smart Fridge/Ambil Biasa - Atas Samping/WIN_20260126_10_17_30_Pro.mp4",
    half=False,
    device="cpu",
    persist=True,
    tracker="hybridsort.yaml",
    stream=True,   # ← WAJIB untuk pipeline yang efisien
    verbose=False,
    save=True,
):
    frame_count += 1
    # Akses hasil di sini kalau perlu, misal:
    # boxes = r.boxes.xyxy  # bounding boxes
    # track_ids = r.boxes.id  # track IDs

t_elapsed = time.perf_counter() - t_start
print(f"⏱  Total time: {t_elapsed:.2f}s  |  avg {t_elapsed / frame_count * 1000:.1f} ms/frame  |  {frame_count / t_elapsed:.1f} FPS  ({frame_count} frames)")