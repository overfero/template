# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

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
        
        # For each track, find the best matching detection to get scores
        for i, track in enumerate(tracks):
            track_box = track[:4]
            
            # Calculate IoU with all detections
            ious = self._calculate_iou(track_box, xyxy)
            
            # Get the detection with highest IoU
            if len(ious) > 0:
                best_match_idx = np.argmax(ious)
                if ious[best_match_idx] > 0.01:  # If there's a reasonable match
                    output_tracks[i, 5] = conf[best_match_idx]  # score
                    output_tracks[i, 6] = cls[best_match_idx]   # class
                    output_tracks[i, 7] = best_match_idx        # detection index
                else:
                    # Use class from HybridSORT if available (column 5)
                    output_tracks[i, 5] = 0.5  # default score
                    output_tracks[i, 6] = tracks[i, 5] if tracks.shape[1] > 5 else 0  # class from HybridSORT
                    output_tracks[i, 7] = i    # default index
            else:
                # Fallback if no matches
                output_tracks[i, 5] = 0.5  # default score
                output_tracks[i, 6] = tracks[i, 5] if tracks.shape[1] > 5 else 0  # class from HybridSORT
                output_tracks[i, 7] = i    # default index
        
        return output_tracks

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