"""Unified inference system for Epee action recognition."""

from .api import load_predictor
from .base import BasePredictor

__all__ = ["load_predictor", "BasePredictor"]