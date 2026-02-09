"""Configuration file for detection and tracking parameters."""

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
HAND_LANDMARKER_MODEL_PATH = str(ULTRALYTICS_ROOT / "checkpoint" / "hand_landmarker.task")
NUM_HANDS = 2

# ============================================================================
# Tracker Configuration
# ============================================================================
USE_DEEPSORT = False  # Set to True to use DeepSort, False to use built-in tracker
DEEPSORT_CONFIG_PATH = str(ULTRALYTICS_ROOT / "trackers" / "deep_sort_pytorch" / "configs" / "deep_sort.yaml")
DEEPSORT_REID_CKPT = str(
    ULTRALYTICS_ROOT / "trackers" / "deep_sort_pytorch" / "deep_sort" / "deep" / "checkpoint" / "ckpt.t7"
)

# ============================================================================
# Camera Configuration
# ============================================================================
CAMERA_FROM_TOP = True  # True = camera from top, False = camera from bottom

# Virtual line position based on camera position
LINE_TOP_CAMERA = [(100, 500), (1800, 500)]  # Line at bottom for top camera
LINE_BOTTOM_CAMERA = [(100, 700), (1800, 700)]  # Line at top for bottom camera

# ============================================================================
# Shelf Detection Lines Configuration
# ============================================================================
# Physical layout from top to bottom:
# - Rak 5: Above line 7-8 (topmost)
# - Rak 4: Between line 7-8 and line 5-6
# - Rak 3: Between line 5-6 and line 3-4
# - Rak 2: Between line 3-4 and line 1-2
# - Rak 1: Below line 1-2 (bottommost)

SHELF_LINE_1_2 = ((330, 938), (1631, 798))  # Line 1-2 (Yellow)
SHELF_LINE_3_4 = ((481, 816), (1500, 730))  # Line 3-4 (Magenta)
SHELF_LINE_5_6 = ((585, 715), (1379, 651))  # Line 5-6 (Cyan)
SHELF_LINE_7_8 = ((665, 634), (1282, 585))  # Line 7-8 (Orange)

# Line colors for visualization
LINE_COLOR_1_2 = (0, 255, 255)  # Yellow
LINE_COLOR_3_4 = (255, 0, 255)  # Magenta
LINE_COLOR_5_6 = (255, 255, 0)  # Cyan
LINE_COLOR_7_8 = (0, 165, 255)  # Orange
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
