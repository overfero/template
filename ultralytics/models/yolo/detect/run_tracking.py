from ultralytics import YOLO
from pathlib import Path

# Model name - will auto-download if not exists
model_name = "yolo26l.pt"

# YOLO class automatically downloads model if not found
# Just like when using CLI: yolo predict model=yolo26n.pt
print(f"Loading model: {model_name}")
if not Path(model_name).exists():
    print(f"Model not found locally, will download from Ultralytics...")

model = YOLO(model_name)  # Auto-downloads if not exists

# Run inference with stream=True (like in docs)
# The custom DetectionPredictor will automatically initialize DeepSort
results = model(
    source="test3.mp4",
    stream=True,  # Use streaming mode to avoid OOM
    save=True,    # Save results with tracking
    show=True,    # Show preview window
    device=0,     # Use GPU 0, or 'cpu' for CPU
    half=True,    # Use FP16 for memory efficiency
    verbose=True  # Show progress
)

print("Processing video with DeepSort tracking...")
print("Press 'q' in the preview window to stop early")
print("Results will be saved to runs/detect/predict*/")
print()

# Process results in streaming mode (like in docs)
frame_count = 0
for result in results:
    frame_count += 1
    
    # You can access result properties if needed
    # boxes = result.boxes  # Boxes with tracking IDs
    # masks = result.masks  # Masks (if using segment model)
    
    if frame_count % 30 == 0:  # Print progress every 30 frames
        print(f"Processed {frame_count} frames...")

print(f"\n✅ Done! Processed {frame_count} frames total.")
print(f"📁 Output saved to: runs/detect/predict*/")
