"""
Package initializer for the `detect` submodule; exposes the public API (e.g. `DetectionPredictor`).

Authors:
    Fehru Madndala Putra (fehruputramen22@gmail.com)
Reviewers:
    Budi Kurniawan (budi.kurniawan1@gdplabs.id)
    Aris Maulana (muhammad.a.maulana@gdplabs.id)
References:
    NONE
"""

# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .predict import DetectionPredictor
__all__ = "DetectionPredictor"
