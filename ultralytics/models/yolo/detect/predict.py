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
    draw_boxes
)
from ultralytics.models.yolo.detect.config import (
    HAND_LANDMARKER_MODEL_PATH, NUM_HANDS,
    USE_DEEPSORT, DEEPSORT_CONFIG_PATH, DEEPSORT_REID_CKPT,
    CAMERA_FROM_TOP, LINE_TOP_CAMERA, LINE_BOTTOM_CAMERA,
    SHELF_LINE_1_2, SHELF_LINE_3_4, SHELF_LINE_5_6, SHELF_LINE_7_8,
    LINE_COLOR_1_2, LINE_COLOR_3_4, LINE_COLOR_5_6, LINE_COLOR_7_8, LINE_COLOR_MAIN,
    LINE_THICKNESS,
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
        
        # Draw custom detection lines
        cv2.line(im0, SHELF_LINE_1_2[0], SHELF_LINE_1_2[1], LINE_COLOR_1_2, LINE_THICKNESS)  # Line 1-2 (Yellow)
        cv2.line(im0, SHELF_LINE_3_4[0], SHELF_LINE_3_4[1], LINE_COLOR_3_4, LINE_THICKNESS)  # Line 3-4 (Magenta)
        cv2.line(im0, SHELF_LINE_5_6[0], SHELF_LINE_5_6[1], LINE_COLOR_5_6, LINE_THICKNESS)  # Line 5-6 (Cyan)
        cv2.line(im0, SHELF_LINE_7_8[0], SHELF_LINE_7_8[1], LINE_COLOR_7_8, LINE_THICKNESS)  # Line 7-8 (Orange)
        
        height, width, _ = im0.shape

        # Display Count - Left: Products Taken, Right: Products Returned
        for idx, (key, value) in enumerate(object_counter.items()):
            cnt_str = str(key) + ":" +str(value)
            cv2.line(im0, (UI_LEFT_MARGIN, UI_TOP_MARGIN), (UI_BOX_WIDTH, UI_TOP_MARGIN), UI_BOX_COLOR, UI_LINE_HEIGHT)
            cv2.putText(im0, f'Numbers of Products Taken', (11, 35), 0, 1, UI_TEXT_COLOR, thickness=UI_TEXT_THICKNESS, lineType=cv2.LINE_AA)    
            cv2.line(im0, (UI_LEFT_MARGIN, 65 + (idx * UI_LINE_HEIGHT)), (UI_BOX_WIDTH, 65 + (idx * UI_LINE_HEIGHT)), UI_BOX_COLOR, 30)
            cv2.putText(im0, cnt_str, (11, 75 + (idx * UI_LINE_HEIGHT)), 0, 1, UI_TEXT_COLOR, thickness=UI_TEXT_THICKNESS, lineType=cv2.LINE_AA)

        for idx, (key, value) in enumerate(object_counter1.items()):
            cnt_str1 = str(key) + ":" +str(value)
            cv2.line(im0, (width - 600, UI_TOP_MARGIN), (width - UI_RIGHT_MARGIN, UI_TOP_MARGIN), UI_BOX_COLOR, UI_LINE_HEIGHT)
            cv2.putText(im0, f'Number of Products Returned', (width - 600, 35), 0, 1, UI_TEXT_COLOR, thickness=UI_TEXT_THICKNESS, lineType=cv2.LINE_AA)
            cv2.line(im0, (width - 600, 65 + (idx * UI_LINE_HEIGHT)), (width - UI_RIGHT_MARGIN, 65 + (idx * UI_LINE_HEIGHT)), UI_BOX_COLOR, 30)
            cv2.putText(im0, cnt_str1, (width - 600, 75 + (idx * UI_LINE_HEIGHT)), 0, 1, [255, 255, 255], thickness=UI_TEXT_THICKNESS, lineType=cv2.LINE_AA)
    
        
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
                    
                    draw_boxes(im0, bbox_xyxy, self.model.names, object_id, identities, data_deque, object_counter, object_counter1, line)
        else:
            # Use built-in Ultralytics tracker
            if hasattr(result, 'boxes') and result.boxes.id is not None:
                # Get tracking IDs from built-in tracker
                bbox_xyxy = result.boxes.xyxy.cpu().numpy()
                identities = result.boxes.id.cpu().numpy().astype(int)
                object_id = result.boxes.cls.cpu().numpy().astype(int)
                
                if len(bbox_xyxy) > 0:
                    draw_boxes(im0, bbox_xyxy, self.model.names, object_id, identities, data_deque, object_counter, object_counter1, line)
            else:
                # No tracking, just draw detections
                self.plotted_img = result.plot()
                im0 = self.plotted_img
        
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

