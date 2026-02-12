# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any
from collections import Counter

import numpy as np

from .basetrack import BaseTrack
from .byte_tracker import BYTETracker
from .hybrid_sort.hybrid_sort import Hybrid_Sort as HybridSort_


class HybridSORT(BYTETracker):
    """HybridSORT tracker wrapper for Ultralytics YOLO framework.
    
    An extended version of BYTETracker that uses the HybridSORT algorithm combining
    SORT with improved association strategies, velocity prediction, and multi-stage matching.
    
    This class wraps the original HybridSORT implementation to be compatible with Ultralytics YOLO's
    tracking pipeline, converting between Ultralytics Results format and HybridSORT's expected format.
    
    Attributes:
        hybrid_sort (HybridSort_): The underlying HybridSORT tracker instance.
        args (Any): Parsed command-line arguments containing tracking parameters.
        frame_id (int): Current frame number.
    
    Methods:
        update: Update tracker with new detections from Ultralytics Results.
        reset: Reset the tracker to its initial state.
        
    Examples:
        Initialize HybridSORT and process detections
        >>> tracker = HybridSORT(args, frame_rate=30)
        >>> results = yolo_model.detect(image)
        >>> tracked_objects = tracker.update(results, image)
    
    Notes:
        HybridSORT improves upon SORT and ByteTrack by using 4-point velocity prediction
        and trajectory-based confidence modulation for better tracking performance.
    """

    def __init__(self, args: Any, frame_rate: int = 30):
        """Initialize HybridSORT tracker with configuration parameters.
        
        Args:
            args (Any): Configuration namespace containing tracking parameters:
                - det_thresh (float): Detection confidence threshold
                - max_age (int): Maximum frames to keep lost tracks
                - min_hits (int): Minimum hits to start a track
                - iou_threshold (float): IoU threshold for matching
                - delta_t (int): Time steps for velocity estimation
                - asso_func (str): Association function ('iou', 'giou', 'ciou', 'diou', 'ct_dist', 'Height_Modulated_IoU')
                - inertia (float): Inertia weight for velocity-based prediction
                - use_byte (bool): Enable ByteTrack-style second association
                - TCM_first_step (bool): Enable Trajectory Confidence Modulation in first matching step
                - TCM_byte_step (bool): Enable Trajectory Confidence Modulation in byte matching step
                - TCM_byte_step_weight (float): Weight for TCM in byte step
            frame_rate (int): Frame rate of the video sequence.
        """
        # Initialize parent BYTETracker (for compatibility)
        super().__init__(args, frame_rate)
        
        # Set default values for HybridSORT-specific parameters
        if not hasattr(args, 'TCM_first_step'):
            args.TCM_first_step = True
        if not hasattr(args, 'TCM_first_step_weight'):
            args.TCM_first_step_weight = 0.5
        if not hasattr(args, 'TCM_byte_step'):
            args.TCM_byte_step = True
        if not hasattr(args, 'TCM_byte_step_weight'):
            args.TCM_byte_step_weight = 0.5
        if not hasattr(args, 'track_thresh'):
            args.track_thresh = 0.5
        
        # Initialize the core HybridSORT tracker
        self.hybrid_sort = HybridSort_(
            args=args,
            det_thresh=getattr(args, 'det_thresh', 0.25),
            max_age=getattr(args, 'max_age', 30),
            min_hits=getattr(args, 'min_hits', 3),
            iou_threshold=getattr(args, 'iou_threshold', 0.3),
            delta_t=getattr(args, 'delta_t', 3),
            asso_func=getattr(args, 'asso_func', 'iou'),
            inertia=getattr(args, 'inertia', 0.2),
            use_byte=getattr(args, 'use_byte', False)
        )
        
        self.args = args
        self.frame_id = 0
        
        # Lost tracks management
        self.lost_tracks = {}  # {track_id: {'last_frame': frame_id, 'cls': class_id}}
        
        # Track appearance history for filtering noise
        self.track_history = {}  # {track_id: [frame_ids where it appeared]}
        
        # Track last known class for each track ID
        self.track_last_class = {}  # {track_id: class_id}

        # Track class history for each track ID (most recent classes seen)
        # {track_id: [class_id1, class_id2, ...]}
        self.track_class_history = {}

    def update(self, results, img: np.ndarray | None = None, feats: np.ndarray | None = None) -> np.ndarray:
        """Update tracker with new detections and return tracked objects.
        
        This method converts Ultralytics Results to HybridSORT format, runs the tracking algorithm,
        and converts the output back to Ultralytics format.
        
        Args:
            results: Ultralytics Results object containing:
                - xyxy: Bounding boxes in (x1, y1, x2, y2) format
                - conf: Detection confidence scores
                - cls: Class labels
            img (np.ndarray, optional): Current frame image (used for img_info).
            feats (np.ndarray, optional): Feature vectors (not used in HybridSORT).
        
        Returns:
            (np.ndarray): Tracked objects in format [x1, y1, x2, y2, track_id, score, cls, idx]
                Shape: (N, 8) where N is the number of tracked objects.
        
        Examples:
            >>> results = model.predict(frame)
            >>> tracks = tracker.update(results[0], frame)
            >>> for track in tracks:
            ...     x1, y1, x2, y2, track_id, score, cls, idx = track
        """
        self.frame_id += 1
        
        # Get image info for HybridSORT - must be plain Python list/tuple with int values
        if img is not None:
            h, w = img.shape[:2]
            img_info = [int(h), int(w)]  # Convert to Python int, not numpy int
            img_size = [int(h), int(w)]  # HybridSORT expects same format
        else:
            # Fallback if no image provided
            img_info = [640, 640]
            img_size = [640, 640]
        
        # Handle empty detections
        if len(results) == 0:
            # Call HybridSORT update with empty array to maintain tracker state
            self.hybrid_sort.update(np.empty((0, 5)), img_info, img_size)
            return np.empty((0, 8))
        
        # Convert Ultralytics Results to HybridSORT format
        # Handle both tensor and numpy array inputs - make copies to avoid in-place modifications
        xyxy = results.xyxy.cpu().numpy().copy() if hasattr(results.xyxy, 'cpu') else np.array(results.xyxy, copy=True)
        conf = results.conf.cpu().numpy().copy() if hasattr(results.conf, 'cpu') else np.array(results.conf, copy=True)
        cls = results.cls.cpu().numpy().copy() if hasattr(results.cls, 'cpu') else np.array(results.cls, copy=True)
        
        # Prepare detections in HybridSORT format: [x1, y1, x2, y2, score]
        # Ensure all data is contiguous numpy arrays with proper dtype
        detections = np.ascontiguousarray(
            np.concatenate([xyxy, conf[:, None]], axis=1), 
            dtype=np.float64
        )
        
        # Run HybridSORT tracking
        # HybridSORT expects img_info and img_size, returns: [x1, y1, x2, y2, track_id, cls, frame_offset]
        tracks = self.hybrid_sort.update(detections, img_info, img_size)
        
        # Handle no tracks returned
        if len(tracks) == 0:
            return np.empty((0, 8))
        
        # Convert HybridSORT output to Ultralytics format
        # HybridSORT returns: [x1, y1, x2, y2, track_id, cls, frame_offset] (7 columns)
        # Ultralytics needs: [x1, y1, x2, y2, track_id, score, cls, idx]
        
        # Create output array
        output_tracks = np.zeros((len(tracks), 8))
        output_tracks[:, :5] = tracks[:, :5]  # [x1, y1, x2, y2, track_id]
        
        # Track current frame's track IDs
        current_track_ids = set()
        
        # For each track, find the best matching detection to get scores and class
        for i, track in enumerate(tracks):
            track_box = track[:4]
            track_id = int(track[4])
            current_track_ids.add(track_id)
            
            # Get class from detection by finding best IoU match
            # (We need this for ID recovery logic)
            track_cls = -1
            if len(cls) > 0:
                ious = self._calculate_iou(track_box, xyxy)
                if len(ious) > 0:
                    best_match_idx = np.argmax(ious)
                    if ious[best_match_idx] > 0.01:
                        track_cls = int(cls[best_match_idx])
            
            # Check if this track was previously lost and is now re-appearing
            # Match based on: same class + ID difference <= 3
            matched_lost_id = None
            min_id_diff = float('inf')
            
            for lost_id, lost_info in self.lost_tracks.items():
                if lost_id not in current_track_ids:  # Don't re-use IDs already in use
                    # Check if class matches
                    if track_cls == lost_info['cls']:
                        # Check if ID difference is <= 3
                        id_diff = abs(track_id - lost_id)
                        if id_diff <= np.inf and id_diff < min_id_diff:
                            min_id_diff = id_diff
                            matched_lost_id = lost_id
            
            # If matched with lost track, use the old ID
            if matched_lost_id is not None:
                output_tracks[i, 4] = matched_lost_id  # Override with old track_id
                
                # CRITICAL: Also update the internal tracker's ID so it persists in future frames
                # Find the tracker object that corresponds to this track
                for trk in self.hybrid_sort.trackers:
                    if trk.id + 1 == track_id:  # +1 because HybridSORT adds 1 to the ID
                        trk.id = matched_lost_id - 1  # -1 because it will be +1 again in output
                        break
                
                # Remove from lost tracks since it's now found
                del self.lost_tracks[matched_lost_id]
                current_track_ids.add(matched_lost_id)
                current_track_ids.discard(track_id)  # Remove new ID
                track_id = matched_lost_id  # Update for history tracking
            
            # Update track history
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append(self.frame_id)
            
            # Calculate IoU with all detections
            ious = self._calculate_iou(track_box, xyxy)
            
            # Get the detection with highest IoU
            if len(ious) > 0:
                best_match_idx = np.argmax(ious)
                if ious[best_match_idx] > 0.01:  # If there's a reasonable match
                    output_tracks[i, 5] = conf[best_match_idx]  # score
                    output_tracks[i, 7] = best_match_idx        # detection index
                    raw_class = int(cls[best_match_idx])
                else:
                    # Use class from HybridSORT if available (column 5)
                    output_tracks[i, 5] = 0.5  # default score
                    raw_class = int(track_cls) if track_cls != -1 else 0
                    output_tracks[i, 7] = i    # default index
            else:
                # Fallback if no matches
                output_tracks[i, 5] = 0.5  # default score
                raw_class = int(track_cls) if track_cls != -1 else 0
                output_tracks[i, 7] = i    # default index

            # Update class history for this track (store raw per-frame class)
            if track_id not in self.track_class_history:
                self.track_class_history[track_id] = []
            self.track_class_history[track_id].append(int(raw_class))
            # Keep history bounded to last 50 entries
            if len(self.track_class_history[track_id]) > 50:
                self.track_class_history[track_id].pop(0)

            # Determine displayed class as the mode of the last up to 5 classes
            last_classes = self.track_class_history[track_id][-5:]
            try:
                display_class = int(Counter(last_classes).most_common(1)[0][0])
            except Exception:
                display_class = int(raw_class)

            output_tracks[i, 6] = display_class
        
        # Store last known class for all current tracks
        for i, track in enumerate(output_tracks):
            track_id = int(track[4])
            class_id = int(track[6])
            self.track_last_class[track_id] = class_id

        # If any lost_tracks IDs are present in the current frame, they have reappeared — remove them
        for tid in list(self.lost_tracks.keys()):
            if tid in current_track_ids:
                try:
                    del self.lost_tracks[tid]
                except KeyError:
                    pass
        
        # Update lost_tracks: Add tracks that disappeared and meet the appearance criteria
        # Only add tracks with at least 5 appearances in the last 15 frames
        previous_track_ids = set(self.lost_tracks.keys()) | set(self.track_history.keys())
        missing_track_ids = previous_track_ids - current_track_ids
        
        for track_id in missing_track_ids:
            if track_id in self.track_history:
                # Check appearance count in last 15 frames
                recent_frames = [f for f in self.track_history[track_id] if self.frame_id - f <= 15]
                if len(recent_frames) >= 5:
                    # This track has appeared at least 5 times in last 15 frames
                    # Get the last known class
                    # Prefer mode of last up to 5 classes seen for stability
                    if track_id in self.track_class_history and len(self.track_class_history[track_id]) > 0:
                        last_classes = self.track_class_history[track_id][-5:]
                        try:
                            track_cls = int(Counter(last_classes).most_common(1)[0][0])
                        except Exception:
                            track_cls = int(self.track_last_class.get(track_id, self.lost_tracks.get(track_id, {}).get('cls', 0)))
                    elif track_id in self.track_last_class:
                        track_cls = self.track_last_class[track_id]
                    elif track_id in self.lost_tracks:
                        track_cls = self.lost_tracks[track_id]['cls']
                    else:
                        track_cls = 0  # fallback
                    
                    self.lost_tracks[track_id] = {
                        'last_frame': self.frame_id,
                        'cls': track_cls
                    }
        
        # Store class info for current tracks (for lost_tracks reference)
        for i, track in enumerate(output_tracks):
            track_id = int(track[4])
            if track_id in self.lost_tracks:
                self.lost_tracks[track_id]['cls'] = int(track[6])

        # Debug: Write tracking info to debug.txt
        self._debug_write_tracking_info(output_tracks)

        return output_tracks

    def _debug_write_tracking_info(self, output_tracks: np.ndarray) -> None:
        """Write current frame tracking info to debug.txt for debugging.

        Logs current objects, lost tracks with frames-since-lost and frames-until-forget,
        and a summary of recent appearance counts per track.
        """
        try:
            with open("debug.txt", "a") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"FRAME {self.frame_id}\n")
                f.write(f"{'='*80}\n\n")
                # Current frame objects
                f.write("CURRENT FRAME OBJECTS:\n")
                current_objects = {}
                for i, track in enumerate(output_tracks):
                    track_id = int(track[4])
                    class_id = int(track[6])
                    score = float(track[5])
                    current_objects[track_id] = {'class': class_id, 'score': score}
                    f.write(f"  Track ID {track_id}: class={class_id}, score={score:.3f}\n")
                if not current_objects:
                    f.write("  (no objects)\n")
                f.write(f"\nCurrent objects dict: {current_objects}\n")

                # Lost tracks
                f.write(f"\n{'─'*60}\n")
                f.write("LOST TRACKS:\n")
                if self.lost_tracks:
                    for track_id, info in self.lost_tracks.items():
                        f.write(f"  Track ID {track_id}: class={info['cls']}\n")
                    f.write(f"\nLost tracks dict: {self.lost_tracks}\n")
                else:
                    f.write("  (no lost tracks)\n")

                # Track history summary
                f.write(f"\n{'─'*60}\n")
                f.write("TRACK HISTORY (last 15 frames):\n")
                if self.track_history:
                    for track_id, frames in self.track_history.items():
                        recent_frames = [fr for fr in frames if self.frame_id - fr <= 15]
                        if recent_frames:
                            f.write(f"  Track ID {track_id}: {len(recent_frames)} appearances in last 15 frames\n")
                else:
                    f.write("  (no history)\n")

                f.write("\n")
        except Exception:
            # avoid crashing tracker on logging errors
            pass

    def _calculate_iou(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Calculate IoU between a single box and multiple boxes.
        
        Args:
            box (np.ndarray): Single bounding box [x1, y1, x2, y2]
            boxes (np.ndarray): Multiple bounding boxes, shape (N, 4)
        
        Returns:
            (np.ndarray): IoU values, shape (N,)
        """
        # Calculate intersection
        xx1 = np.maximum(box[0], boxes[:, 0])
        yy1 = np.maximum(box[1], boxes[:, 1])
        xx2 = np.minimum(box[2], boxes[:, 2])
        yy2 = np.minimum(box[3], boxes[:, 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h
        
        # Calculate union
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = box_area + boxes_area - intersection
        
        # Calculate IoU
        iou = intersection / (union + 1e-6)
        return iou

    def reset(self):
        """Reset the HybridSORT tracker to its initial state.
        
        Clears all tracked objects and resets frame counter.
        
        Examples:
            >>> tracker = HybridSORT(args, frame_rate=30)
            >>> # ... track some frames ...
            >>> tracker.reset()  # Start fresh for new video
        """
        # Reset HybridSORT internal state
        self.hybrid_sort.trackers = []
        self.hybrid_sort.frame_count = 0
        
        # Reset frame counter
        self.frame_id = 0
        
        # Reset track ID counter
        from .hybrid_sort.hybrid_sort import KalmanBoxTracker
        KalmanBoxTracker.count = 0
        
        # Also reset parent class state
        super().reset()

    @staticmethod
    def reset_id():
        """Reset the ID counter for track instances.
        
        This ensures unique track IDs across tracking sessions.
        
        Examples:
            >>> HybridSORT.reset_id()  # Reset global track ID counter
        """
        from .hybrid_sort.hybrid_sort import KalmanBoxTracker
        KalmanBoxTracker.count = 0
        BaseTrack.reset_id()