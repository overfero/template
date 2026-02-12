from ultralytics import YOLO
from pathlib import Path


model_name = "ultralytics/checkpoint/best1.pt"

#
print(f"Loading model: {model_name}")
if not Path(model_name).exists():
    print(f"Model not found locally, will download from Ultralytics...")

model = YOLO(model_name)

results = model.track(
    source="/home/overfero/Project/glair/Jumpstart - Smart Fridge/Ambil Biasa - Bawah/WIN_20260126_10_45_33_Pro.mp4",
    stream=True,  
    save=True,    
    show=False,    
    device=0,     
    persist=True,  
    tracker="hybridsort.yaml"  # Changed to OC-SORT tracker
)

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
