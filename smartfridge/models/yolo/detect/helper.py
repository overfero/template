#@markdown We implemented some functions to visualize the hand landmark detection results. <br/> Run the following cell to activate the functions.
import mediapipe as mp
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

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def draw_landmarks_on_image(rgb_image, detection_result):
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


def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
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


def draw_border(img, pt1, pt2, color, thickness, r, d):
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


def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def is_point_below_line(point, line_start, line_end):
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


def is_point_above_line(point, line_start, line_end):
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


def get_direction(point1, point2):
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


def draw_boxes(img, stored_object, identities, object_counter, object_counter1, line, current_frame):
    # iterate over a static list of keys so we can safely modify the dict
    for key in list(stored_object.keys()):
        prod = stored_object[key]
        if prod.taken_counted and prod.return_counted:
            stored_object.pop(key)

    for i, id in enumerate(identities):
        if id not in stored_object.keys():
            continue
        prod = stored_object[id]
        x1, y1, x2, y2 = [int(i) for i in prod.bbox]

        # center: bottom-middle for top camera, top-middle for bottom camera
        if CAMERA_FROM_TOP:
            center = (int((x1 + x2) / 2), int(y2))
        else:
            center = (int((x1 + x2) / 2), int(y1 * 0.8))

        color = compute_color_for_labels(prod.class_id)
        obj_name = prod.class_name
        label = f"{obj_name}"

        prod.trail_points.appendleft(center)
        # check for crossing only when we have at least two trail points
        if len(prod.trail_points) >= 2:
            p0 = tuple(map(int, prod.trail_points[0]))
            p1 = tuple(map(int, prod.trail_points[1]))
            direction = get_direction(p0, p1)
            prod.movement_direction = direction
            if intersect(p0, p1, line[0], line[1]):
                cv2.line(img, line[0], line[1], (255, 255, 255), 3)
                
                # Logic based on camera position
                if "North" in direction:
                    if not prod.taken_counted:
                        obj_label = f"{obj_name}"
                        if obj_label not in object_counter:
                            object_counter[obj_label] = 1
                        else:
                            object_counter[obj_label] += 1
                        prod.taken_counted = True
                        prod.last_seen_frame = current_frame

                if "South" in direction:
                    # Only count returned if the product was previously counted as taken
                    if prod.taken_counted and not prod.return_counted:
                        obj_label = obj_name
                        if obj_label not in object_counter1:
                            object_counter1[obj_label] = 1
                        else:
                            object_counter1[obj_label] += 1
                        prod.return_counted = True
                        prod.last_seen_frame = current_frame

        UI_box(prod.bbox, img, label=label, color=color, line_thickness=2)

    # with open("track.txt", "a") as f:
    #     for obj_id, prod in stored_object.items():
    #         # Format: frame, id, class_name, bbox, taken_counted, return_counted
    #         lin = f"{current_frame},{obj_id},{prod.class_name},{prod.taken_counted},{prod.return_counted}, {prod.movement_direction}\n"
    #         f.write(lin)
    
    return img