#!/usr/bin/env python3
"""
Run YOLO detection with DeepSort tracking using stream mode.
Usage: python run_tracking.py

This script demonstrates how to use custom DetectionPredictor with DeepSort
in streaming mode, just like the official Ultralytics API example.
"""

from ultralytics import YOLO

# Load model - will automatically use our custom DetectionPredictor from predict.py
# which includes DeepSort tracking
model = YOLO("yolo26l.pt")

# Run inference with stream=True (like in docs)
# The custom DetectionPredictor will automatically initialize DeepSort
results = model(
    source="test3.mp4",
    stream=True,  # Use streaming mode to avoid OOM
    save=True,    # Save results with tracking
    show=True,    # Show preview window
    imgsz=640,    # Smaller image size for memory efficiency
    conf=0.25,    # Confidence threshold
    iou=0.45,     # IoU threshold
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
print(f"🎯 Video includes: DeepSort tracking, speed estimation, vehicle counting")
