"""
Product data model for tracked objects: trail management, counting flags, simple serialization
helpers and movement metadata used by the tracker.

Authors:
    Fehru Madndala Putra (fehruputramen22@gmail.com)
Reviewers:
    Budi Kurniawan (budi.kurniawan1@gdplabs.id)
    Aris Maulana (muhammad.a.maulana@gdplabs.id)
References:
    NONE
"""

from collections import deque
from typing import Tuple, List, Optional


class Product:
    """
    Represents a tracked product/object with movement information.
    
    Attributes:
        id (int): Unique tracking ID for this product
        class_id (int): Class ID from the detection model
        class_name (str): Human-readable class name
        current_position (Tuple[int, int]): Current (x, y) position
        trail_points (deque): Deque of historical positions
        total_displacement (float): Total distance moved
        is_below_line (bool): Whether product is below virtual line
        movement_direction (str): Direction of movement (North/South/East/West)
        taken_counted (bool): Whether this product has been counted as taken
        return_counted (bool): Whether this product has been counted as returned
        last_seen_frame (int): Last frame number where this product was detected
    """
    
    def __init__(
        self,
        id: int,
        class_id: int,
        class_name: str,
        current_position: Tuple[int, int],
        bbox: List[int],
        trail_points: Optional[List[Tuple[int, int]]] = None,
        is_below_line: bool = False,
        movement_direction: str = "",
        taken_counted: bool = False,
        return_counted: bool = False,
        last_seen_frame: int = 0
    ):
        """
        Initialize a Product instance.
        
        Args:
            id: Unique tracking ID
            class_id: Detection model class ID
            class_name: Human-readable class name
            current_position: Current (x, y) position
            trail_points: List of historical positions (optional)
            bbox: Bounding box coordinates
            is_below_line: Whether below virtual line (default: False)
            movement_direction: Movement direction string (default: "")
            taken_counted: Whether counted as taken (default: False)
            return_counted: Whether counted as returned (default: False)
            last_seen_frame: Last frame number detected (default: 0)
        """
        self.id = id
        self.class_id = class_id
        self.class_name = class_name
        self.current_position = current_position
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.trail_points = deque(trail_points or [], maxlen=64)
        self.total_displacement = 0.0
        self.is_below_line = is_below_line
        self.movement_direction = movement_direction
        self.taken_counted = taken_counted
        self.return_counted = return_counted
        self.last_seen_frame = last_seen_frame
    
    def to_dict(self) -> dict:
        """
        Convert Product to dictionary format for compatibility.
        
        Returns:
            Dictionary with product information
        """
        return {
            'id': self.id,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'current_position': self.current_position,
            'trail_points': list(self.trail_points),
            'total_displacement': self.total_displacement,
            'is_below_line': self.is_below_line,
            'movement_direction': self.movement_direction,
            'taken_counted': self.taken_counted,
            'return_counted': self.return_counted,
            'last_seen_frame': self.last_seen_frame
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Product':
        """
        Create Product instance from dictionary.
        
        Args:
            data: Dictionary with product information
            
        Returns:
            Product instance
        """
        return cls(
            id=data['id'],
            class_id=data['class_id'],
            class_name=data['class_name'],
            current_position=data['current_position'],
            trail_points=data.get('trail_points', []),
            total_displacement=data.get('total_displacement', 0.0),
            is_below_line=data.get('is_below_line', False),
            movement_direction=data.get('movement_direction', ''),
            taken_counted=data.get('taken_counted', False),
            return_counted=data.get('return_counted', False),
            last_seen_frame=data.get('last_seen_frame', 0)
        )
    
    def merge_trail(self, other_trail: List[Tuple[int, int]]) -> None:
        """
        Merge another trail into this product's trail.
        
        Args:
            other_trail: List of trail points to merge
        """
        # Combine new trail with existing trail
        combined = list(other_trail) + list(self.trail_points)
        self.trail_points = deque(combined, maxlen=64)
    
    def __repr__(self) -> str:
        """String representation of Product."""
        return (f"Product(id={self.id}, class='{self.class_name}', "
                f"pos={self.current_position}, displacement={self.total_displacement:.2f})")
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.class_name} #{self.id} at {self.current_position}"
