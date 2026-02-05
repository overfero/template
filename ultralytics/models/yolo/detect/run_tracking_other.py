#!/usr/bin/env python3
"""
Run YOLO detection with ByteTrack tracking (built-in Ultralytics tracker).
Usage: python run_tracking_bytetrack.py

ByteTrack is the default built-in tracker in Ultralytics YOLO.
It's faster and lighter than DeepSort, but may have less accurate re-identification.
"""

from ultralytics import YOLO

# Load model
model = YOLO("yolo26l.pt")

# Run inference with ByteTrack (built-in tracker)
results = model.track(  # Use .track() instead of .predict() for tracking
    source="test3.mp4",
    stream=True,        # Use streaming mode to avoid OOM
    save=True,          # Save results with tracking
    show=True,          # Show preview window
    device=0,           # Use GPU 0, or 'cpu' for CPU
    half=True,          # Use FP16 for memory efficiency
    # tracker="bytetrack.yaml",  # Use ByteTrack tracker (default)
    tracker="botsort.yaml",  # Alternative: use BotSort tracker
    verbose=True,       # Show progress
    persist=True,       # Persist tracks between frames
)

print("🚀 Processing video with ByteTrack (built-in tracker)...")
print("📺 Press 'q' in the preview window to stop early")
print("💾 Results will be saved to runs/detect/track*/")
print()

# Process results in streaming mode
frame_count = 0
for result in results:
    frame_count += 1
    
    # Access tracking information
    if result.boxes is not None and result.boxes.id is not None:
        # Get tracking IDs
        track_ids = result.boxes.id.cpu().numpy().astype(int)
        # Get boxes
        boxes = result.boxes.xyxy.cpu().numpy()
        # Get confidence scores
        confs = result.boxes.conf.cpu().numpy()
        # Get class IDs
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        # Example: print info for first detection
        if len(track_ids) > 0 and frame_count % 30 == 0:
            print(f"Frame {frame_count}: {len(track_ids)} objects tracked")
    
    if frame_count % 30 == 0:  # Print progress every 30 frames
        print(f"Processed {frame_count} frames...")

print(f"\n✅ Done! Processed {frame_count} frames total.")
print(f"📁 Output saved to: runs/detect/track*/")

