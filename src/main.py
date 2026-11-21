import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.sam3_inference.inference import Sam3Segmenter
from src.sam3_inference.iou import calculate_iou
from src.sam3_inference.visualization import draw_detections

LOGGER = logging.getLogger(__name__)

data = json.load(open("datasets/annotations/Test.json", encoding="utf-8"))
