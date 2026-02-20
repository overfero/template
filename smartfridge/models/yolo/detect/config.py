"""
Configuration constants for MediaPipe hand detection, camera layout, UI and model defaults
used by the `detect` pipeline.

Authors:
    Fehru Madndala Putra (fehruputramen22@gmail.com)
Reviewers:
    Budi Kurniawan (budi.kurniawan1@gdplabs.id)
    Aris Maulana (muhammad.a.maulana@gdplabs.id)
References:
    NONE
"""

from pathlib import Path
from typing import Tuple, List

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
# Camera Configuration
# ============================================================================
CAMERA_FROM_TOP = True  # True = camera from top, False = camera from bottom

# Virtual line position based on camera position
LINE_TOP_CAMERA = [(50, 400), (1800, 300)]
LINE_BOTTOM_CAMERA = [(0, 100), (2000, 100)]  #

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
UI_BOX_COLOR: Tuple[int, int, int] = (85, 45, 255)
UI_TEXT_COLOR: Tuple[int, int, int] = (225, 255, 255)
UI_TEXT_THICKNESS = 2

# ============================================================================
# Model Configuration
# ============================================================================
DEFAULT_FPS = 30  # Assuming 30fps for timestamp calculation

# Convenience computed values
# Virtual line based on camera orientation
LINE = LINE_TOP_CAMERA if CAMERA_FROM_TOP else LINE_BOTTOM_CAMERA

# Consolidated UI configuration to allow a single import
UI_CONFIG = {
	"UI_LEFT_MARGIN": UI_LEFT_MARGIN,
	"UI_RIGHT_MARGIN": UI_RIGHT_MARGIN,
	"UI_TOP_MARGIN": UI_TOP_MARGIN,
	"UI_LINE_HEIGHT": UI_LINE_HEIGHT,
	"UI_BOX_WIDTH": UI_BOX_WIDTH,
	"UI_BOX_COLOR": UI_BOX_COLOR,
	"UI_TEXT_COLOR": UI_TEXT_COLOR,
	"UI_TEXT_THICKNESS": UI_TEXT_THICKNESS,
	"LINE_COLOR_MAIN": LINE_COLOR_MAIN,
	"LINE_THICKNESS": LINE_THICKNESS,
}


