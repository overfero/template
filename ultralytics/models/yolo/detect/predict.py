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

# Global state variables
data_deque = {}
deepsort = None
object_counter = {}
object_counter1 = {}
moving_objects = None
# Store mapping of old_id -> new_id for objects that reappeared
id_mapping = {}
# Store original moving Product instances by class name
stored_moving_objects = {}  # {class_name: Product}
# Track IDs that have been counted to prevent duplicate counting
# Changed from set to dict to track last crossing direction (North/South)
# This allows same ID to be counted for both taken (North) and returned (South)
counted_crossing_ids = {}  # {ID: 'North' or 'South'} - Track last crossing direction
# Track IDs that were seen below the line
ids_below_line = set()  # IDs detected below virtual line
ids_above_line = set()  # IDs detected above virtual line

# Set virtual line position based on camera position from config
line = LINE_TOP_CAMERA if CAMERA_FROM_TOP else LINE_BOTTOM_CAMERA

def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file(DEEPSORT_CONFIG_PATH)
    
    # Override REID_CKPT with absolute path from config
    cfg_deep.DEEPSORT.REID_CKPT = DEEPSORT_REID_CKPT

    deepsort= DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True)


class DetectionPredictor(BasePredictor):
    """A class extending the BasePredictor class for prediction based on a detection model.

    This predictor specializes in object detection tasks, processing model outputs into meaningful detection results
    with bounding boxes and class predictions.

    Attributes:
        args (namespace): Configuration arguments for the predictor.
        model (nn.Module): The detection model used for inference.
        batch (list): Batch of images and metadata for processing.

    Methods:
        postprocess: Process raw model predictions into detection results.
        construct_results: Build Results objects from processed predictions.
        construct_result: Create a single Result object from a prediction.
        get_obj_feats: Extract object features from the feature maps.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.detect import DetectionPredictor
        >>> args = dict(model="yolo26n.pt", source=ASSETS)
        >>> predictor = DetectionPredictor(overrides=args)
        >>> predictor.predict_cli()
    """
    
    def setup_model(self, model, verbose=True):
        """Initialize tracker when setting up the model."""
        global deepsort
        if USE_DEEPSORT and deepsort is None:
            init_tracker()
        super().setup_model(model, verbose)

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))


    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Post-process predictions and return a list of Results objects.

        This method applies non-maximum suppression to raw model predictions and prepares them for visualization and
        further analysis.

        Args:
            preds (torch.Tensor): Raw predictions from the model.
            img (torch.Tensor): Processed input image tensor in model input format.
            orig_imgs (torch.Tensor | list): Original input images before preprocessing.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            (list): List of Results objects containing the post-processed predictions.

        Examples:
            >>> predictor = DetectionPredictor(overrides=dict(model="yolo26n.pt"))
            >>> results = predictor.predict("path/to/image.jpg")
            >>> processed_results = predictor.postprocess(preds, img, orig_imgs)
        """
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
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Calculate timestamp in milliseconds
        timestamp_ms = int(frame * 1000 / DEFAULT_FPS) if frame is not None else 0
        
        # Detect hand landmarks
        try:
            detection_result = detector.detect_for_video(mp_image, timestamp_ms)
            
            # Draw landmarks on RGB frame
            annotated_rgb = draw_landmarks_on_image(rgb_frame, detection_result)
            
            # Convert back to BGR for OpenCV
            im0 = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            # If hand detection fails, continue with original image
            print(f"Hand detection error: {e}")
            pass
        
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
        
        if USE_DEEPSORT:
            # DeepSort tracking
            xywh_bboxs = []
            confs = []
            oids = []
            
            for *xyxy, conf, cls in det:
                x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                xywh_bboxs.append(xywh_obj)
                confs.append([conf.item()])
                oids.append(int(cls))
            
            if len(xywh_bboxs) > 0:
                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)
                
                outputs = deepsort.update(xywhs, confss, oids, im0)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    object_id = outputs[:, -1]
                    
                    draw_boxes(im0, bbox_xyxy, self.model.names, object_id, identities, data_deque, object_counter, object_counter1, line, counted_crossing_ids, stored_moving_objects, frame if frame is not None else 0)
        else:
            # Use built-in Ultralytics tracker
            if hasattr(result, 'boxes') and result.boxes.id is not None:
                # Get tracking IDs from built-in tracker
                bbox_xyxy = result.boxes.xyxy.cpu().numpy()
                identities = result.boxes.id.cpu().numpy().astype(int)
                object_id = result.boxes.cls.cpu().numpy().astype(int)
                
                # Apply ID mapping persistently (map new IDs back to original IDs)
                for idx in range(len(identities)):
                    if identities[idx] in id_mapping:
                        identities[idx] = id_mapping[identities[idx]]
                
                if len(bbox_xyxy) > 0:
                    # Track objects position relative to line
                    for bbox, identity, obj_class_id in zip(bbox_xyxy, identities, object_id):
                        center_x = int((bbox[0] + bbox[2]) / 2)
                        center_y = int((bbox[1] + bbox[3]) / 2)
                        current_pos = (center_x, center_y)
                        identity = int(identity)
                        obj_class_id = int(obj_class_id)
                        class_name = self.model.names[obj_class_id]
                        
                        # Track if object is below line
                        if is_point_below_line(current_pos, line[0], line[1]):
                            ids_below_line.add(identity)

                            # If we have stored Product info and it was previously counted as taken,
                            # mark it as returned immediately when detected below the line.
                            product = stored_moving_objects.get(class_name)
                            if product and product.id == identity and getattr(product, 'taken_counted', False) and not getattr(product, 'returned_counted', False):
                                obj_label = f"{class_name}"
                                if obj_label not in object_counter1:
                                    object_counter1[obj_label] = 0
                                object_counter1[obj_label] += 1
                                product.returned_counted = True
                                counted_crossing_ids[identity] = 'South'
                                if identity in ids_above_line:
                                    ids_above_line.discard(identity)
                                print(f"[AUTO-RETURNED] {class_name} ID {identity} detected below line and was previously taken - Returned count: {object_counter1[obj_label]}")

                        # If object was below line and now above line = taken (if not already counted as North)
                        elif is_point_above_line(current_pos, line[0], line[1]) and identity in ids_below_line:
                            last_direction = counted_crossing_ids.get(identity, None)
                            
                            # Count as taken if: never counted OR last was returned (South)
                            # Prefer Product-level flag if available
                            product = stored_moving_objects.get(class_name)
                            if product and product.id == identity:
                                # Only increment when product hasn't been marked as taken
                                if not product.taken_counted:
                                    obj_label = f"{class_name}"
                                    if obj_label not in object_counter:
                                        object_counter[obj_label] = 0
                                    object_counter[obj_label] += 1
                                    product.taken_counted = True
                                    if identity in ids_below_line:
                                        ids_below_line.discard(identity)
                                    print(f"[TAKEN] {class_name} ID {identity} moved from below to above line - Taken count: {object_counter[obj_label]}")
                                else:
                                    # already counted at product level; skip
                                    pass
                            else:
                                # Fallback: no Product info available, use last_direction logic
                                if last_direction is None or last_direction == 'South':
                                    obj_label = f"{class_name}"
                                    if obj_label not in object_counter:
                                        object_counter[obj_label] = 0
                                    object_counter[obj_label] += 1
                                    counted_crossing_ids[identity] = 'North'
                                    # remove from below-line set to avoid duplicate counting
                                    if identity in ids_below_line:
                                        ids_below_line.discard(identity)
                                    print(f"[TAKEN] {class_name} ID {identity} moved from below to above line - Taken count: {object_counter[obj_label]}")


                        if is_point_above_line(current_pos, line[0], line[1]):
                            ids_above_line.add(identity)
                        
                        # If object was above line and now below line = returned (if not already counted as South)
                        elif is_point_below_line(current_pos, line[0], line[1]) and identity in ids_above_line:
                            last_direction = counted_crossing_ids.get(identity, None)
                            
                            # Count as returned if: never counted OR last was taken (North)
                            # Prefer Product-level flag if available
                            product = stored_moving_objects.get(class_name)
                            if product and product.id == identity:
                                # Only increment when product hasn't been marked as returned
                                if not product.returned_counted:
                                    obj_label = f"{class_name}"
                                    if obj_label not in object_counter1:
                                        object_counter1[obj_label] = 0
                                    object_counter1[obj_label] += 1
                                    product.returned_counted = True
                                    counted_crossing_ids[identity] = 'South'
                                    # remove from above-line set to avoid duplicate counting
                                    if identity in ids_above_line:
                                        ids_above_line.discard(identity)
                                    print(f"[RETURNED] {class_name} ID {identity} moved from above to below line - Returned count: {object_counter1[obj_label]}")
                                else:
                                    # already counted at product level; skip
                                    pass
                            else:
                                # Fallback: no Product info available, use last_direction logic
                                if last_direction is None or last_direction == 'North':
                                    obj_label = f"{class_name}"
                                    if obj_label not in object_counter1:
                                        object_counter1[obj_label] = 0
                                    object_counter1[obj_label] += 1
                                    counted_crossing_ids[identity] = 'South'
                                    # remove from above-line set to avoid duplicate counting
                                    if identity in ids_above_line:
                                        ids_above_line.discard(identity)
                                    print(f"[RETURNED] {class_name} ID {identity} moved from above to below line - Returned count: {object_counter1[obj_label]}")
                    
                    # Now draw boxes with potentially updated identities
                    draw_boxes(im0, bbox_xyxy, self.model.names, object_id, identities, data_deque, object_counter, object_counter1, line, counted_crossing_ids, stored_moving_objects, frame if frame is not None else 0)
            else:
                # No tracking, just draw detections
                self.plotted_img = result.plot()
                im0 = self.plotted_img
        
        
        # Write per-frame trace including state (taken_counted/returned_counted) per id
        try:
            trace_path = self.save_dir.parent / "trace.txt" if hasattr(self, "save_dir") else "trace.txt"
            if hasattr(self, "save_dir"):
                self.save_dir.parent.mkdir(parents=True, exist_ok=True)

            def _get_product_flags(identity):
                for cname, prod in stored_moving_objects.items():
                    if prod is not None and getattr(prod, 'id', None) == identity:
                        return cname, bool(getattr(prod, 'taken_counted', False)), bool(getattr(prod, 'returned_counted', False))
                return None, False, False

            below_info = []
            for iid in sorted(ids_below_line):
                cname, taken_f, returned_f = _get_product_flags(iid)
                below_info.append({"id": int(iid), "class": cname, "taken_counted": taken_f, "returned_counted": returned_f})

            above_info = []
            for iid in sorted(ids_above_line):
                cname, taken_f, returned_f = _get_product_flags(iid)
                above_info.append({"id": int(iid), "class": cname, "taken_counted": taken_f, "returned_counted": returned_f})

            with open(str(trace_path), "a", encoding="utf-8") as tf:
                tf.write(f"Frame {frame if frame is not None else 0}: below={below_info}, above={above_info}\n")
        except Exception:
            pass

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

