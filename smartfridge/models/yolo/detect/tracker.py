from typing import Dict, Sequence, Optional, Tuple
from collections import deque

from smartfridge.models.yolo.detect.product import Product
from smartfridge.models.yolo.detect.helper import (
    get_or_update_product,
    process_crossing_for_product,
    update_line_membership_and_counts,
)


class Tracker:
    """Minimal tracker that encapsulates stored objects, membership sets, and counters.

    This class delegates per-object updates to helper functions in
    `smartfridge.models.yolo.detect.helper`.
    """

    def __init__(self):
        self.stored_objects: Dict[int, Product] = {}
        self.ids_below_line: set = set()
        self.ids_above_line: set = set()
        self.taken_counter: Dict[str, int] = {}
        self.returned_counter: Dict[str, int] = {}

    def update_with_detections(
        self,
        bbox_xyxy,
        identities,
        class_ids,
        frame: Optional[int],
        model_names: Sequence[str],
        line: Tuple[Tuple[int, int], Tuple[int, int]],
    ) -> None:
        """Update tracker state given current frame detections.

        Args:
            bbox_xyxy: iterable of [x1,y1,x2,y2] boxes (numpy array or list)
            identities: iterable of int ids (same length as bbox_xyxy)
            class_ids: iterable of int class ids (same length)
            frame: current frame index or None
            model_names: mapping from class id to class name
            line: virtual line tuple used for crossing logic
        """
        for bbox, identity, obj_class_id in zip(bbox_xyxy, identities, class_ids):
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)

            product = get_or_update_product(
                self.stored_objects,
                int(identity),
                int(obj_class_id),
                bbox,
                (center_x, center_y),
                frame,
                model_names,
            )

            # maintain trail
            try:
                product.trail_points.appendleft((center_x, center_y))
            except (AttributeError, TypeError):
                product.trail_points = deque([(center_x, center_y)])

            # crossing-based counting (trail intersects line)
            taken_delta, returned_delta, _ = process_crossing_for_product(product, line, frame)
            for k, v in taken_delta.items():
                self.taken_counter[k] = self.taken_counter.get(k, 0) + v
            for k, v in returned_delta.items():
                self.returned_counter[k] = self.returned_counter.get(k, 0) + v

            # Update membership sets and counters
            update_line_membership_and_counts(
                product,
                int(identity),
                self.ids_below_line,
                self.ids_above_line,
                self.taken_counter,
                self.returned_counter,
                line,
            )

        # remove completed products
        for key in list(self.stored_objects.keys()):
            prod = self.stored_objects[key]
            if prod.taken_counted and prod.return_counted:
                self.stored_objects.pop(key, None)
