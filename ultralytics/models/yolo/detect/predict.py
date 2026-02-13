# Ultralytics YOLO 🚀, GPL-3.0 license
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import re
import cv2
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops, nms
from ultralytics.utils.plotting import Annotator

from ultralytics.trackers.deep_sort_pytorch.utils.parser import get_config
from ultralytics.trackers.deep_sort_pytorch.deep_sort import DeepSort
from ultralytics.models.yolo.detect.helper import (
    draw_landmarks_on_image, 
    xyxy_to_xywh, 
    draw_boxes,
    is_point_above_line,
    is_point_below_line
)
from ultralytics.models.yolo.detect.product import Product
from ultralytics.models.yolo.detect.config import (
    HAND_LANDMARKER_MODEL_PATH, NUM_HANDS,
    USE_DEEPSORT, DEEPSORT_CONFIG_PATH, DEEPSORT_REID_CKPT,
    CAMERA_FROM_TOP, LINE_TOP_CAMERA, LINE_BOTTOM_CAMERA,
    LINE_COLOR_MAIN, LINE_THICKNESS,
    UI_LEFT_MARGIN, UI_RIGHT_MARGIN, UI_TOP_MARGIN, UI_LINE_HEIGHT,
    UI_BOX_WIDTH, UI_BOX_COLOR, UI_TEXT_COLOR, UI_TEXT_THICKNESS,
    DEFAULT_FPS,
)

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

# Initialize MediaPipe Hand Landmarker with config values
base_options = python.BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=NUM_HANDS,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

object_counter = {}
object_counter1 = {}
ids_below_line = set()  # IDs detected below virtual line
ids_above_line = set()  # IDs detected above virtual line
stored_moving_objects = {}

# Set virtual line position based on camera position from config
line = LINE_TOP_CAMERA if CAMERA_FROM_TOP else LINE_BOTTOM_CAMERA


class DetectionPredictor(BasePredictor):
    """YOLO Detection Predictor for inference and tracking."""

    def get_annotator(self, img):
        """Get annotator object for image."""
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))


    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Postprocess model predictions: apply NMS and construct Results objects."""
        save_feats = getattr(self, "_feats", None) is not None
        preds = nms.non_max_suppression(
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
    
    def write_results(self, i, p, im, s):
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
        im0 = result.orig_img.copy()
        
        # HAND LANDMARK DETECTION
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(frame * 1000 / DEFAULT_FPS) if frame is not None else 0
        
        # Detect hand landmarks
        # try:
        #     detection_result = detector.detect_for_video(mp_image, timestamp_ms)
        #     annotated_rgb = draw_landmarks_on_image(rgb_frame, detection_result)
        #     im0 = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        # except Exception as e:
        #     print(f"Hand detection error: {e}")
        #     pass
        
        # Get predictions for this image
        det = result.boxes.data  # xyxy, conf, cls

        # Draw reference lines
        cv2.line(im0, line[0], line[1], LINE_COLOR_MAIN, LINE_THICKNESS)
    
        height, width, _ = im0.shape

        # Display net Taken counts (Taken - Returned). Show only non-zero nets.
        net_counts = {}
        for k, v in object_counter.items():
            net_counts[k] = v - object_counter1.get(k, 0)
        for k, v in object_counter1.items():
            if k not in net_counts:
                net_counts[k] = -v

        displayed = [(k, cnt) for k, cnt in net_counts.items() if cnt != 0]
        if displayed:
            cv2.line(im0, (UI_LEFT_MARGIN, UI_TOP_MARGIN), (UI_BOX_WIDTH, UI_TOP_MARGIN), UI_BOX_COLOR, UI_LINE_HEIGHT)
            cv2.putText(im0, f'Products Taken (net)', (11, 35), 0, 1, UI_TEXT_COLOR, thickness=UI_TEXT_THICKNESS, lineType=cv2.LINE_AA)
        for idx, (key, value) in enumerate(displayed):
            cnt_str = f"{key}:{value}"
            cv2.line(im0, (UI_LEFT_MARGIN, 65 + (idx * UI_LINE_HEIGHT)), (UI_BOX_WIDTH, 65 + (idx * UI_LINE_HEIGHT)), UI_BOX_COLOR, 30)
            cv2.putText(im0, cnt_str, (11, 75 + (idx * UI_LINE_HEIGHT)), 0, 1, UI_TEXT_COLOR, thickness=UI_TEXT_THICKNESS, lineType=cv2.LINE_AA)
    
        # Display frame number at bottom right
        frame_text = f"Frame: {frame if frame is not None else 0}"
        text_size = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = width - text_size[0] - 15
        text_y = height - 15
        cv2.putText(im0, frame_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        
        if len(det) == 0:
            string += f"{result.verbose()}{result.speed['inference']:.1f}ms"
            # Set plotted image with line and counters even when no detections
            self.plotted_img = im0
            if self.args.save_txt:
                result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
            if self.args.save_crop:
                result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)
            if self.args.show:
                self.show(str(p))
            if self.args.save:
                self.save_predicted_images(self.save_dir / p.name, frame)
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
            
            # Track objects position relative to line
            for bbox, identity, obj_class_id in zip(bbox_xyxy, identities, object_id):
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)

                # Reuse existing Product object if present so we preserve trail_points
                # and counting flags. Creating a new Product every frame discards history.
                existing = stored_moving_objects.get(identity)
                if existing is not None:
                    product = existing
                    product.class_id = int(obj_class_id)
                    product.class_name = self.model.names[int(obj_class_id)]
                    product.current_position = (center_x, center_y)
                    product.bbox = bbox.tolist()
                    product.last_seen_frame = frame if frame is not None else 0
                else:
                    product = Product(
                        id=int(identity),
                        class_id=int(obj_class_id),
                        class_name=self.model.names[int(obj_class_id)],
                        current_position=(center_x, center_y),
                        bbox=bbox.tolist(),
                        last_seen_frame=frame if frame is not None else 0,
                    )
                    stored_moving_objects[identity] = product
                
                # Track if object is below line
                if is_point_below_line(product.current_position, line[0], line[1]):
                    ids_below_line.add(identity)

                    if product.taken_counted and not product.return_counted and is_point_below_line(product.trail_points[0], line[0], line[1]):
                        obj_label = f"{product.class_name}"
                        if obj_label not in object_counter1:
                            object_counter1[obj_label] = 0
                        object_counter1[obj_label] += 1
                        product.return_counted = True
                        if identity in ids_above_line:
                            ids_above_line.discard(identity)

                # If object was below line and now above line = taken (if not already counted as North)
                elif is_point_above_line(product.current_position, line[0], line[1]) and identity in ids_below_line:
                    # last_direction = counted_crossing_ids.get(identity, None)
                    if not product.taken_counted:
                        obj_label = f"{product.class_name}"
                        if obj_label not in object_counter:
                            object_counter[obj_label] = 0
                        object_counter[obj_label] += 1
                        product.taken_counted = True
                        if identity in ids_below_line:
                            ids_below_line.discard(identity)
                    else:
                        # already counted at product level; skip
                        pass


                if is_point_above_line(product.current_position, line[0], line[1]):
                    ids_above_line.add(identity)
                
                # If object was above line and now below line = returned (if not already counted as South)
                elif is_point_below_line(product.current_position, line[0], line[1]) and identity in ids_above_line:
                    if product.taken_counted and not product.return_counted:
                        obj_label = f"{product.class_name}"
                        if obj_label not in object_counter1:
                            object_counter1[obj_label] = 0
                        object_counter1[obj_label] += 1
                        product.return_counted = True
                        product.movement_direction = 'South'
                        # remove from above-line set to avoid duplicate counting
                        if identity in ids_above_line:
                            ids_above_line.discard(identity)
                    else:
                        # already counted at product level; skip
                        pass
            
            # Now draw boxes with potentially updated identities
            draw_boxes(im0, stored_moving_objects, identities, object_counter, object_counter1, line, frame)

        # Set plotted image with tracking results
        self.plotted_img = im0
        
        # Save and show with tracking results
        if self.args.save_txt:
            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)
        if self.args.show:
            self.show(str(p))
        if self.args.save:
            self.save_predicted_images(self.save_dir / p.name, frame)
            
        string += f"{result.speed['inference']:.1f}ms"
        return string

    @staticmethod
    def get_obj_feats(feat_maps, idxs):
        """Extract object features from the feature maps."""
        import torch

        s = min(x.shape[1] for x in feat_maps)  # find shortest vector length
        obj_feats = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, s, x.shape[1] // s).mean(dim=-1) for x in feat_maps], dim=1
        )  # mean reduce all vectors to same length
        return [feats[idx] if idx.shape[0] else [] for feats, idx in zip(obj_feats, idxs)]  # for each img in batch

    def construct_results(self, preds, img, orig_imgs):
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

    def construct_result(self, pred, img, orig_img, img_path):
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

