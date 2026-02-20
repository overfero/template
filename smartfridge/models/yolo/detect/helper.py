"""
Helper utilities for the `detect` pipeline: visualization, MediaPipe hand-landmarker integration,
bounding-box conversions, UI rendering and other small helper functions used by tracker/predictor.

Authors:
    Fehru Madndala Putra (fehruputramen22@gmail.com)
Reviewers:
    Budi Kurniawan (budi.kurniawan1@gdplabs.id)
    Aris Maulana (muhammad.a.maulana@gdplabs.id)
References:
    NONE
"""

#@markdown We implemented some functions to visualize the hand landmark detection results. <br/> Run the following cell to activate the functions.
import mediapipe as mp
import logging
import numpy as np
import cv2
from mediapipe.tasks import python
from smartfridge.models.yolo.detect.product import Product
from mediapipe.tasks.python import vision
from numpy import random
from collections import deque
from smartfridge.models.yolo.detect.config import (
    MARGIN, FONT_SIZE, FONT_THICKNESS, HANDEDNESS_TEXT_COLOR, CAMERA_FROM_TOP,
)
from typing import Any, Dict, Tuple, Optional, Sequence
from dataclasses import dataclass

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def draw_landmarks_on_image(rgb_image: np.ndarray, detection_result: Any) -> np.ndarray:
    """Draw MediaPipe hand landmarks on `rgb_image` and return annotated image.

    Args:
        rgb_image: HxWx3 RGB image as numpy array.
        detection_result: result returned by MediaPipe hand landmarker.

    Returns:
        Annotated RGB image as numpy array.
    """
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image


def detect_and_annotate_hands(
    bgr_image: np.ndarray, detector: Any, frame: Optional[int], fps: int
) -> np.ndarray:
    """Run MediaPipe hand landmarker on a BGR image and return annotated BGR image.

    Returns the annotated image on success, or the original image on failure.
    """
    logger = logging.getLogger(__name__)
    try:
        rgb_frame = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(frame * 1000 / fps) if frame is not None else 0
        detection_result = detector.detect_for_video(mp_image, timestamp_ms)
        annotated_rgb = draw_landmarks_on_image(rgb_frame, detection_result)
        return cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
    except (cv2.error, ValueError, RuntimeError) as e:
        # Catch common image/MediaPipe/runtime errors but allow other
        # unexpected exceptions to surface during development.
        logger.exception("Hand detection failed: %s", e)
        return bgr_image


def render_ui(
    img: np.ndarray,
    taken_counter: Dict[str, int],
    returned_counter: Dict[str, int],
    frame: Optional[int],
    line: Tuple[Tuple[int, int], Tuple[int, int]],
    ui_config: "UIConfig",
) -> np.ndarray:
    """Render UI overlays: counters, frame text, and reference line.

    Mutates and returns `img` with drawn UI elements.
    """
    UI_LEFT_MARGIN = ui_config.UI_LEFT_MARGIN
    UI_BOX_WIDTH = ui_config.UI_BOX_WIDTH
    UI_TOP_MARGIN = ui_config.UI_TOP_MARGIN
    UI_LINE_HEIGHT = ui_config.UI_LINE_HEIGHT
    UI_BOX_COLOR = ui_config.UI_BOX_COLOR
    UI_TEXT_COLOR = ui_config.UI_TEXT_COLOR
    UI_TEXT_THICKNESS = ui_config.UI_TEXT_THICKNESS

    # Draw reference line
    cv2.line(img, line[0], line[1], ui_config.LINE_COLOR_MAIN, ui_config.LINE_THICKNESS)

    # Display net Taken counts (Taken - Returned). Show only non-zero nets.
    net_counts = {}
    for k, v in taken_counter.items():
        net_counts[k] = v - returned_counter.get(k, 0)
    for k, v in returned_counter.items():
        if k not in net_counts:
            net_counts[k] = -v

    displayed = [(k, cnt) for k, cnt in net_counts.items() if cnt != 0]
    if displayed:
        cv2.line(img, (UI_LEFT_MARGIN, UI_TOP_MARGIN), (UI_BOX_WIDTH, UI_TOP_MARGIN), UI_BOX_COLOR, UI_LINE_HEIGHT)
        cv2.putText(img, f'Products Taken (net)', (11, 35), 0, 1, UI_TEXT_COLOR, thickness=UI_TEXT_THICKNESS, lineType=cv2.LINE_AA)
    for idx, (key, value) in enumerate(displayed):
        cnt_str = f"{key}:{value}"
        cv2.line(img, (UI_LEFT_MARGIN, 65 + (idx * UI_LINE_HEIGHT)), (UI_BOX_WIDTH, 65 + (idx * UI_LINE_HEIGHT)), UI_BOX_COLOR, 30)
        cv2.putText(img, cnt_str, (11, 75 + (idx * UI_LINE_HEIGHT)), 0, 1, UI_TEXT_COLOR, thickness=UI_TEXT_THICKNESS, lineType=cv2.LINE_AA)

    # Display frame number at bottom right
    height, width = img.shape[:2]
    frame_text = f"Frame: {frame if frame is not None else 0}"
    text_size = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = width - text_size[0] - 15
    text_y = height - 15
    cv2.putText(img, frame_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)

    return img


@dataclass
class UIConfig:
    UI_LEFT_MARGIN: int
    UI_RIGHT_MARGIN: int
    UI_BOX_WIDTH: int
    UI_TOP_MARGIN: int
    UI_LINE_HEIGHT: int
    UI_BOX_COLOR: Sequence[int]
    UI_TEXT_COLOR: Sequence[int]
    UI_TEXT_THICKNESS: int
    LINE_COLOR_MAIN: Sequence[int]
    LINE_THICKNESS: int


def xyxy_to_xywh(*xyxy):
    """Calculate the bounding box center and size from absolute xyxy values.

    Returns (x_c, y_c, w, h) as floats.
    """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def xyxy_to_tlwh(bbox_xyxy: Sequence[Sequence[float]]) -> list:
    """Convert list/array of [x1,y1,x2,y2] boxes to [x,y,w,h] integer format.

    Returns a list of [x, y, w, h] where x,y are the top-left corner.
    """
    tlwh_bboxs: list = []
    for box in bbox_xyxy:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        x = x1
        y = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [x, y, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0: #person
        color = (85,45,255)
    elif label == 2: # Car
        color = (222,82,175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_border(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Sequence[int],
    thickness: int,
    r: int,
    d: int,
) -> np.ndarray:
    """Draw rounded border and decorative corners between `pt1` and `pt2`.

    Returns the modified image (same object as `img`).
    """
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
    
    return img


def UI_box(
    x: Sequence[float],
    img: np.ndarray,
    color: Optional[Sequence[int]] = None,
    label: Optional[str] = None,
    line_thickness: Optional[int] = None,
) -> None:
    """Plot one bounding box with optional label on `img` (mutates in-place)."""
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        draw_border(img, (c1[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0], c1[1] + 3), color, 1, 8, 2)

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def intersect(A: Tuple[int, int], B: Tuple[int, int], C: Tuple[int, int], D: Tuple[int, int]) -> bool:
    """Return True if line segment AB intersects CD (2D)."""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A: Tuple[int, int], B: Tuple[int, int], C: Tuple[int, int]) -> bool:
    """Helper: counter-clockwise test for three 2D points."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def is_point_below_line(point: Tuple[float, float], line_start: Tuple[int, int], line_end: Tuple[int, int]) -> bool:
    """Check if a point is below a line (higher y value in image coordinates)."""
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Calculate y position on the line at point's x coordinate
    if x2 - x1 == 0:  # Vertical line
        return False
    
    # Linear interpolation to find y on the line at x
    line_y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    
    # Point is below if its y is larger (because y increases downward in images)
    return y > line_y


def is_point_above_line(point: Tuple[float, float], line_start: Tuple[int, int], line_end: Tuple[int, int]) -> bool:
    """Check if a point is above a line (lower y value in image coordinates)."""
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Calculate y position on the line at point's x coordinate
    if x2 - x1 == 0:  # Vertical line
        return False
    
    # Linear interpolation to find y on the line at x
    line_y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    
    # Point is above if its y is smaller (because y increases downward in images)
    return y < line_y


def get_direction(point1: Tuple[int, int], point2: Tuple[int, int]) -> str:
    """Return rough cardinal direction between two points (N/S and E/W)."""
    direction_str = ""

    # calculate y axis direction
    if point1[1] > point2[1]:
        direction_str += "South"
    elif point1[1] < point2[1]:
        direction_str += "North"
    else:
        direction_str += ""

    # calculate x axis direction
    if point1[0] > point2[0]:
        direction_str += "East"
    elif point1[0] < point2[0]:
        direction_str += "West"
    else:
        direction_str += ""

    return direction_str


def get_or_update_product(
    stored_objects: Dict[int, Product],
    identity: int,
    obj_class_id: int,
    bbox: Any,
    center: Tuple[int, int],
    frame: Optional[int],
    model_names: Sequence[str],
) -> Product:
    """Fetch existing Product by `identity` or create a new one.

    Ensures `prod.bbox` is a plain Python list (uses `.tolist()` only if available).
    Returns the Product instance stored in `stored_objects`.
    """
    existing = stored_objects.get(identity)
    if existing is not None:
        prod = existing
        prod.class_id = int(obj_class_id)
        prod.class_name = model_names[int(obj_class_id)]
        prod.current_position = (int(center[0]), int(center[1]))
        # ensure bbox is a plain list of numbers
        if hasattr(bbox, "tolist"):
            prod.bbox = bbox.tolist()
        else:
            prod.bbox = list(bbox)
        prod.last_seen_frame = frame if frame is not None else 0
    else:
        prod = Product(
            id=int(identity),
            class_id=int(obj_class_id),
            class_name=model_names[int(obj_class_id)],
            current_position=(int(center[0]), int(center[1])),
            bbox=(bbox.tolist() if hasattr(bbox, "tolist") else list(bbox)),
            last_seen_frame=frame if frame is not None else 0,
        )
        stored_objects[identity] = prod

    return prod

def process_crossing_for_product(
    prod: Product,
    line: Tuple[Tuple[int, int], Tuple[int, int]],
    current_frame: Optional[int],
) -> Tuple[Dict[str, int], Dict[str, int], bool]:
    """Handle trail intersection and counting for a single product.

    Returns a tuple `(taken_delta, returned_delta, intersected_flag)` where the
    dicts map class-name -> delta count. This function mutates `prod` flags
    (e.g. `taken_counted`, `return_counted`, `movement_direction`,
    `last_seen_frame`) but does not draw on any image.
    """
    taken: Dict[str, int] = {}
    returned: Dict[str, int] = {}
    intersected = False

    if len(prod.trail_points) >= 2:
        p0 = tuple(map(int, prod.trail_points[0]))
        p1 = tuple(map(int, prod.trail_points[1]))
        direction = get_direction(p0, p1)
        prod.movement_direction = direction
        if intersect(p0, p1, line[0], line[1]):
            intersected = True

            # Logic based on camera position
            if "North" in direction:
                if not prod.taken_counted:
                    obj_label = f"{prod.class_name}"
                    taken[obj_label] = taken.get(obj_label, 0) + 1
                    prod.taken_counted = True
                    prod.last_seen_frame = current_frame

            if "South" in direction:
                # Only count returned if the product was previously counted as taken
                if prod.taken_counted and not prod.return_counted:
                    obj_label = prod.class_name
                    returned[obj_label] = returned.get(obj_label, 0) + 1
                    prod.return_counted = True
                    prod.last_seen_frame = current_frame

    return taken, returned, intersected


def update_line_membership_and_counts(
    prod: Product,
    identity: int,
    ids_below_line: set,
    ids_above_line: set,
    taken_counter: Dict[str, int],
    returned_counter: Dict[str, int],
    line: Tuple[Tuple[int, int], Tuple[int, int]],
) -> None:
    """Update below/above-line membership sets and counters for one product.

    Mutates `ids_below_line`, `ids_above_line`, `taken_counter`, `returned_counter`,
    and `prod` flags in-place.
    """
    # Track if object is below line
    if is_point_below_line(prod.current_position, line[0], line[1]):
        ids_below_line.add(identity)

        if (
            prod.taken_counted
            and not prod.return_counted
            and len(prod.trail_points)
            and is_point_below_line(prod.trail_points[0], line[0], line[1])
        ):
            obj_label = f"{prod.class_name}"
            returned_counter[obj_label] = returned_counter.get(obj_label, 0) + 1
            prod.return_counted = True
            if identity in ids_above_line:
                ids_above_line.discard(identity)

    # If object was below line and now above line = taken
    elif is_point_above_line(prod.current_position, line[0], line[1]) and identity in ids_below_line:
        if not prod.taken_counted:
            obj_label = f"{prod.class_name}"
            taken_counter[obj_label] = taken_counter.get(obj_label, 0) + 1
            prod.taken_counted = True
            if identity in ids_below_line:
                ids_below_line.discard(identity)

    # Maintain above-line membership
    if is_point_above_line(prod.current_position, line[0], line[1]):
        ids_above_line.add(identity)

    # If object was above line and now below line = returned
    elif is_point_below_line(prod.current_position, line[0], line[1]) and identity in ids_above_line:
        if prod.taken_counted and not prod.return_counted:
            obj_label = f"{prod.class_name}"
            returned_counter[obj_label] = returned_counter.get(obj_label, 0) + 1
            prod.return_counted = True
            prod.movement_direction = "South"
            if identity in ids_above_line:
                ids_above_line.discard(identity)

    # function mutates passed-in collections and product; no return value


def draw_boxes(
    img: np.ndarray,
    stored_object: Dict[int, Product],
    identities: Sequence[int],
    line: Tuple[Tuple[int, int], Tuple[int, int]],
    current_frame: Optional[int],
) -> np.ndarray:
    """Draw UI boxes for tracked objects.

    This function is focused solely on rendering and does not modify product
    counting state or draw the reference line on intersection.
    """
    # remove completed products
    for key in list(stored_object.keys()):
        prod = stored_object[key]
        if prod.taken_counted and prod.return_counted:
            stored_object.pop(key)

    for i, id in enumerate(identities):
        if id not in stored_object:
            continue
        prod = stored_object[id]
        color = compute_color_for_labels(prod.class_id)
        label = f"{prod.class_name}"
        UI_box(prod.bbox, img, label=label, color=color, line_thickness=2)

    return img