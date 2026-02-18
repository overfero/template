# Ultralytics YOLO 🚀, GPL-3.0 license
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import re
import cv2
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Any, List, Sequence, Tuple, Optional, Union
from pathlib import Path
import numpy as np

from smartfridge.engine.predictor import BasePredictor
from smartfridge.engine.results import Results
from smartfridge.utils import ops
from smartfridge.utils.plotting import Annotator

from smartfridge.models.yolo.detect.helper import (
    draw_boxes,
    detect_and_annotate_hands,
    render_ui,
    UIConfig,
)
from smartfridge.models.yolo.detect.tracker import Tracker
from smartfridge.models.yolo.detect.product import Product
from smartfridge.models.yolo.detect.config import (
    HAND_LANDMARKER_MODEL_PATH,
    NUM_HANDS,
    CAMERA_FROM_TOP,
    LINE_TOP_CAMERA,
    LINE_BOTTOM_CAMERA,
    UI_CONFIG,
    DEFAULT_FPS,
)

# Set virtual line position based on camera position from config
line = LINE_TOP_CAMERA if CAMERA_FROM_TOP else LINE_BOTTOM_CAMERA


class DetectionPredictor(BasePredictor):
    """YOLO Detection Predictor for inference and tracking."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # encapsulated state (use Tracker instance for shared state)
        self.line = line
        # lazy MediaPipe detector (created on first use)
        self._detector = None

        # pre-create UI config to avoid per-frame allocation
        # Use consolidated UI_CONFIG from config.py for simpler imports
        self.ui_cfg = UIConfig(**UI_CONFIG)

        # tracker encapsulates stored objects and counters
        self.tracker = Tracker()

    def _get_detector(self) -> Any:
        """Lazily initialize and return the MediaPipe HandLandmarker instance.

        Returns:
            MediaPipe HandLandmarker instance (opaque type).
        """
        if self._detector is None:
            base_options = python.BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL_PATH)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=NUM_HANDS,
                running_mode=vision.RunningMode.VIDEO,
            )
            self._detector = vision.HandLandmarker.create_from_options(options)
        return self._detector

    def close(self) -> None:
        """Release resources held by the predictor (e.g. MediaPipe detector)."""
        det = getattr(self, "_detector", None)
        if det is not None:
            try:
                det.close()
            except Exception:
                pass
            self._detector = None

    def get_annotator(self, img: np.ndarray) -> Annotator:
        """Get annotator object for image.

        Args:
            img: BGR image as numpy array.
        Returns:
            An `Annotator` instance for drawing on `img`.
        """
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def _save_and_show(self, p: Path, frame: Optional[int], result: Results) -> None:
        """Save text/crops and optionally show or save plotted images for a frame.

        This consolidates the repeated block used in `_handle_frame`.
        """
        if self.args.save_txt:
            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)
        if self.args.show:
            self.show(str(p))
        if self.args.save:
            self.save_predicted_images(self.save_dir / p.name, frame)


    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Postprocess model predictions: apply NMS and construct Results objects."""
        save_feats = getattr(self, "_feats", None) is not None
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=0 if self.args.task == "detect" else len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
            return_idxs=save_feats,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        if save_feats:
            obj_feats = self.get_obj_feats(self._feats, preds[1])
            preds = preds[0]

        results = self.construct_results(preds, img, orig_imgs, **kwargs)

        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f  # add object features to results

        return results
    
    def postprocess(self, preds: Sequence[torch.Tensor], img: torch.Tensor, orig_imgs: Union[Sequence[np.ndarray], np.ndarray], **kwargs: Any) -> List[Results]:
        """Postprocess model predictions: apply NMS and construct Results objects.

        Args:
            preds: Sequence of model prediction tensors.
            img: Preprocessed batch tensor used for inference.
            orig_imgs: Original images (numpy arrays) before preprocessing.
        Returns:
            List of `Results` objects.
        """

        save_feats = getattr(self, "_feats", None) is not None
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=0 if self.args.task == "detect" else len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
            return_idxs=save_feats,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        if save_feats:
            obj_feats = self.get_obj_feats(self._feats, preds[1])
            preds = preds[0]

        results = self.construct_results(preds, img, orig_imgs, **kwargs)

        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f  # add object features to results

        return results


    def write_results(self, i: int, p: Path, im: np.ndarray, s: Sequence[str]) -> str:
        """Write detection results with tracking."""
        string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        
        # Determine if streaming/webcam source
        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:
            string += f"{i}: "
            frame = self.dataset.count
        else:
            match = re.search(r"frame (\d+)/", s[i])
            frame = int(match[1]) if match else None

        self.txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))
        string += "{:g}x{:g} ".format(*im.shape[2:])

        result = self.results[i]
        result.save_dir = self.save_dir.__str__()
        # Delegate per-frame processing to helper method
        string += self._handle_frame(i, p, im, s, result, frame)
        return string

    def _handle_frame(self, i: int, p: Path, im: np.ndarray, s: Sequence[str], result: Results, frame: Optional[int]) -> str:
        """Process a single frame: detection annotation, tracking, counting, drawing and saving.

        Returns a string suffix describing detections and timing (same as original write_results).
        """
        string = ""
        im0 = result.orig_img.copy()

        # HAND LANDMARK DETECTION and UI rendering (use lazy detector)
        im0 = detect_and_annotate_hands(im0, self._get_detector(), frame, DEFAULT_FPS)
        det = result.boxes.data  # xyxy, conf, cls

        # Render UI overlays (reference line, counters, frame) using Tracker counters
        im0 = render_ui(im0, self.tracker.taken_counter, self.tracker.returned_counter, frame, self.line, self.ui_cfg)

        if len(det) == 0:
            string += f"{result.verbose()}{result.speed['inference']:.1f}ms"
            # Set plotted image with line and counters even when no detections
            self.plotted_img = im0
            self._save_and_show(p, frame, result)
            return string

        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()
            string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        # Use built-in Ultralytics tracker
        if hasattr(result, 'boxes') and result.boxes.id is not None:
            # Get tracking IDs from built-in tracker
            bbox_xyxy = result.boxes.xyxy.cpu().numpy()
            identities = result.boxes.id.cpu().numpy().astype(int)
            object_id = result.boxes.cls.cpu().numpy().astype(int)

            # Let Tracker handle per-object bookkeeping (trail, counters, membership)
            self.tracker.update_with_detections(
                bbox_xyxy, identities, object_id, frame, self.model.names, self.line
            )

            # Now draw boxes (drawing only) using Tracker's stored objects
            im0 = draw_boxes(im0, self.tracker.stored_objects, identities, self.line, frame)

        # Set plotted image with tracking results
        self.plotted_img = im0

        # Save and show with tracking results
        self._save_and_show(p, frame, result)

        string += f"{result.speed['inference']:.1f}ms"
        return string

    @staticmethod
    def get_obj_feats(feat_maps: Sequence[torch.Tensor], idxs: Sequence[torch.Tensor]) -> List[Any]:
        """Extract object features from the feature maps.

        Args:
            feat_maps: Sequence of feature map tensors.
            idxs: Index tensors selecting objects per image.
        Returns:
            List of per-image feature tensors/lists.
        """
        import torch

        s = min(x.shape[1] for x in feat_maps)  # find shortest vector length
        obj_feats = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, s, x.shape[1] // s).mean(dim=-1) for x in feat_maps], dim=1
        )  # mean reduce all vectors to same length
        return [feats[idx] if idx.shape[0] else [] for feats, idx in zip(obj_feats, idxs)]  # for each img in batch

    def construct_results(self, preds: Sequence[torch.Tensor], img: torch.Tensor, orig_imgs: Sequence[np.ndarray]) -> List[Results]:
        """Construct a list of Results objects from model predictions.

        Args:
            preds (list[torch.Tensor]): List of predicted bounding boxes and scores for each image.
            img (torch.Tensor): Batch of preprocessed images used for inference.
            orig_imgs (list[np.ndarray]): List of original images before preprocessing.

        Returns:
            (list[Results]): List of Results objects containing detection information for each image.
        """
        return [
            self.construct_result(pred, img, orig_img, img_path)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]

    def construct_result(self, pred: torch.Tensor, img: torch.Tensor, orig_img: np.ndarray, img_path: str) -> Results:
        """Construct a single Results object from one image prediction.

        Args:
            pred (torch.Tensor): Predicted boxes and scores with shape (N, 6) where N is the number of detections.
            img (torch.Tensor): Preprocessed image tensor used for inference.
            orig_img (np.ndarray): Original image before preprocessing.
            img_path (str): Path to the original image file.

        Returns:
            (Results): Results object containing the original image, image path, class names, and scaled bounding boxes.
        """
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])

