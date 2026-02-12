"""
Configuration file for detection and tracking parameters.
"""

from pathlib import Path

# Get the directory where this config file is located
CONFIG_DIR = Path(__file__).resolve().parent
# Get ultralytics root directory (3 levels up from detect folder)
ULTRALYTICS_ROOT = CONFIG_DIR.parent.parent.parent

# ============================================================================
# MediaPipe Hand Detection Configuration
# ============================================================================
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green
HAND_LANDMARKER_MODEL_PATH = str(ULTRALYTICS_ROOT / 'checkpoint' / 'hand_landmarker.task')
NUM_HANDS = 2

# ============================================================================
# Tracker Configuration
# ============================================================================
USE_DEEPSORT = True  # Set to True to use DeepSort, False to use built-in tracker
DEEPSORT_CONFIG_PATH = str(ULTRALYTICS_ROOT / "trackers" / "deep_sort_pytorch" / "configs" / "deep_sort.yaml")
DEEPSORT_REID_CKPT = str(ULTRALYTICS_ROOT / "trackers" / "deep_sort_pytorch" / "deep_sort" / "deep" / "checkpoint" / "ckpt.t7")

# ============================================================================
# Camera Configuration
# ============================================================================
CAMERA_FROM_TOP = False  # True = camera from top, False = camera from bottom

# Virtual line position based on camera position
LINE_TOP_CAMERA = [(100, 500), (1800, 500)]  # Line at bottom for top camera
LINE_BOTTOM_CAMERA = [(0, 100), (2000, 100)]  # Line at top for bottom camera

# Line colors for visualization
LINE_COLOR_MAIN = (46, 162, 112)  # Main line color

# Line thickness
LINE_THICKNESS = 3

# ============================================================================
# UI Configuration
# ============================================================================
UI_LEFT_MARGIN = 20
UI_RIGHT_MARGIN = 30
UI_TOP_MARGIN = 25
UI_LINE_HEIGHT = 40
UI_BOX_WIDTH = 500
UI_BOX_COLOR = [85, 45, 255]
UI_TEXT_COLOR = [225, 255, 255]
UI_TEXT_THICKNESS = 2

# ============================================================================
# Model Configuration
# ============================================================================
DEFAULT_FPS = 30  # Assuming 30fps for timestamp calculation

# ============================================================================
# File Paths
# ============================================================================
# Output file path - will be created in current working directory
SHELF_COORDINATES_FILE = "shelf_coordinates.txt"
