"""Paths, images, annotator records, folds and run-length encoding helpers.

Every training image can carry several independent annotator records (record id "TTGGAA-<stem>"). The metric scores a
prediction against each record separately, so annotations are always kept per record and never merged.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as cocomask

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("FILAMENT_DATA", ROOT / "data"))      # the unpacked Kaggle dataset
WORK = Path(os.environ.get("FILAMENT_WORK", ROOT / "work"))      # everything the pipeline writes
TRAIN_IMAGES = DATA / "train/train_images"
TEST_IMAGES = DATA / "test/test_images"
ANNOTATIONS = DATA / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
SIZE = 2048                                                      # every image is 2048 x 2048


def load_image(stem, test=False):
    with Image.open((TEST_IMAGES if test else TRAIN_IMAGES) / f"{stem}.jpeg") as image:
        return np.asarray(image.convert("L")).copy()


def train_stems():
    return sorted(folds())


def test_stems():
    return sorted(p.stem for p in TEST_IMAGES.glob("*.jpeg"))


@lru_cache(maxsize=None)
def folds():
    """stem -> fold 0..3. Folds are whole calendar months, so near-duplicate days never straddle folds."""
    return json.loads((ROOT / "folds.json").read_text())


def fold_stems(fold):
    return sorted(s for s, k in folds().items() if k == fold)


@lru_cache(maxsize=None)
def records():
    """stem -> list of annotator records (sorted by id), each {"id", "rles"}: one full-canvas COCO RLE per filament."""
    raw = json.loads(ANNOTATIONS.read_text())
    by_record = {}
    for a in raw["annotations"]:
        by_record.setdefault(a["image_id"], []).append(cocomask.merge(cocomask.frPyObjects(a["segmentation"], SIZE, SIZE)))
    out = {}
    for image in sorted(raw["images"], key=lambda i: i["id"]):
        out.setdefault(Path(image["file_name"]).stem, []).append({"id": image["id"], "rles": by_record.get(image["id"], [])})
    return out


def encode(mask, x0=0, y0=0):
    """Full-canvas RLE of a boolean crop whose top-left corner sits at (x0, y0)."""
    canvas = np.zeros((SIZE, SIZE), np.uint8, order="F")
    canvas[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]] = mask
    return cocomask.encode(canvas)


def decode_window(rle):
    """Boolean crop of an RLE's bounding box and its top-left corner: (mask, x0, y0)."""
    x, y, w, h = (int(round(v)) for v in cocomask.toBbox(rle))
    return cocomask.decode(rle)[y:y + h, x:x + w].astype(bool), x, y


def largest_component(mask):
    """The largest 8-connected component of a boolean mask, or None if the mask is empty."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
