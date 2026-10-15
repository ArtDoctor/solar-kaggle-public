"""Filament detector: YOLO26s boxes, trained on native-resolution tiles, run on the full image in eight views.

Training data: every training image is cut into 3 x 3 tiles of 1024 px (stride 512). Each (annotator record, tile) pair
is one sample whose labels are the polygons of that annotator's filaments visible in the tile, so the detector sees every
annotator's opinion. Polygons rather than boxes keep the boxes tight under rotation augmentation.

Inference: the full 2048 px image at two input sizes (2048 and 1280) times four flips. The eight box sets are fused by
weighted box fusion; a fused score is the mean over the eight views (a view without the box counts as 0).
"""
import hashlib
import json
import shutil
from multiprocessing import Pool

import cv2
import numpy as np
from pycocotools import mask as cocomask

from .data import SIZE, WORK, fold_stems, load_image, records, test_stems, train_stems

TILE = 1024
ORIGINS = [(x, y) for y in (0, 512, 1024) for x in (0, 512, 1024)]
SIZES = (2048, 1280)
FLIPS = {"id": (False, False), "lr": (True, False), "ud": (False, True), "lrud": (True, True)}
TRAIN_ARGS = dict(imgsz=TILE, batch=8, epochs=12, workers=6, device=0, seed=42, deterministic=False, optimizer="AdamW",
                  lr0=0.001, lrf=0.02, cos_lr=True, amp=True, warmup_epochs=1.0, single_cls=True, max_det=300,
                  patience=100, plots=False, hsv_h=0.0, hsv_s=0.0, hsv_v=0.2, mosaic=0.0, mixup=0.0, scale=0.25,
                  translate=0.1, degrees=15.0, fliplr=0.5, flipud=0.5, erasing=0.0)


def detector_path(fold):
    return WORK / "weights" / f"detector_fold{fold}.pt"


def proposals_path(fold, split):
    """split "heldout" = the fold's own training images (out of fold), "test" = the test images."""
    return WORK / "proposals" / f"{split}_fold{fold}.jsonl"


# ----------------------------------------------------------------------------------------------------------- training

def tile_polygon(mask, x0, y0, min_px=30, min_fraction=0.25):
    """Normalised polygon of the part of a filament mask inside the tile at (x0, y0), or None if too little is visible."""
    crop = mask[y0:y0 + TILE, x0:x0 + TILE]
    visible = int(crop.sum())
    if visible < min_px or visible < min_fraction * int(mask.sum()):
        return None
    contours, _ = cv2.findContours(crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c.reshape(-1, 2) for c in contours if len(c) >= 3]
    ys, xs = np.nonzero(crop)
    if not contours:
        contours = [np.array([[xs.min(), ys.min()], [xs.max() + 1, ys.min()], [xs.max() + 1, ys.max() + 1], [xs.min(), ys.max() + 1]])]
    points = []
    for contour in contours:                      # several pieces are chained into one polygon
        pts = contour[::max(1, len(contour) // 60)].astype(float)
        points.extend(pts.tolist() + [pts[0].tolist()])
    points.append(points[0])
    arr = np.asarray(points) + 0.5                # pixel centres -> pixel edges, clipped to the visible extent
    arr[:, 0] = np.clip(arr[:, 0], xs.min(), xs.max() + 1)
    arr[:, 1] = np.clip(arr[:, 1], ys.min(), ys.max() + 1)
    return (arr / TILE).clip(0, 1)


def keep_empty(tile):
    """Tiles without filaments are kept 30% of the time, chosen by a hash so that every run keeps the same ones."""
    return int(hashlib.md5(tile.encode()).hexdigest(), 16) % 100 < 30


def _tile_image(stem):
    image = load_image(stem)
    masks = {r["id"]: [cocomask.decode(rle).astype(bool) for rle in r["rles"]] for r in records()[stem]}
    rows = []
    for x0, y0 in ORIGINS:
        tile = f"{stem}__{x0}_{y0}"
        path = WORK / "tiles/images" / f"{tile}.png"
        if not path.exists():
            cv2.imwrite(str(path), image[y0:y0 + TILE, x0:x0 + TILE])
        for record_id, record_masks in masks.items():
            polygons = [p for p in (tile_polygon(m, x0, y0) for m in record_masks) if p is not None]
            if not polygons and not keep_empty(tile):
                continue
            sample = f"{record_id}__{x0}_{y0}"
            (WORK / "tiles/labels" / f"{sample}.txt").write_text(
                "".join("0 " + " ".join(f"{v:.6f}" for v in np.round(p, 6).reshape(-1)) + "\n" for p in polygons))
            rows.append({"stem": stem, "sample": sample, "tile": tile, "n": len(polygons)})
    return rows


def build_tiles(workers=5):
    """Tile every training image once, then link one YOLO dataset per fold (training images = all other folds)."""
    (WORK / "tiles/images").mkdir(parents=True, exist_ok=True)
    (WORK / "tiles/labels").mkdir(parents=True, exist_ok=True)
    records()                                     # parse the annotations once, before the workers fork
    with Pool(workers) as pool:
        rows = [row for chunk in pool.imap_unordered(_tile_image, train_stems(), chunksize=4) for row in chunk]
    for fold in range(4):
        root = WORK / "detector" / f"fold{fold}"
        if root.exists():
            shutil.rmtree(root)
        own = set(fold_stems(fold))
        monitor = set(sorted({r["stem"] for r in rows} - own)[::25])  # a few training images to watch the loss
        for row in rows:
            if row["stem"] in own:
                continue
            split = "val" if row["stem"] in monitor and row["sample"].endswith("_512_512") and row["n"] else "train"
            for kind, source, suffix in (("images", WORK / "tiles/images" / f"{row['tile']}.png", "png"),
                                         ("labels", WORK / "tiles/labels" / f"{row['sample']}.txt", "txt")):
                link = root / kind / split / f"{row['sample']}.{suffix}"
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(source)
        (root / "data.yaml").write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\nnc: 1\n")
        print(f"fold {fold}: {len(list((root / 'images/train').iterdir()))} training samples", flush=True)


def train(fold):
    """Fine-tune the COCO-pretrained YOLO26s on all folds but `fold`; the last epoch is kept."""
    from ultralytics import YOLO
    run = WORK / "detector/runs" / f"fold{fold}"
    if run.exists():
        shutil.rmtree(run)
    YOLO("yolo26s.pt").train(data=str(WORK / "detector" / f"fold{fold}" / "data.yaml"), project=str(run.parent),
                             name=run.name, **TRAIN_ARGS)
    detector_path(fold).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(run / "weights/last.pt", detector_path(fold))


# ---------------------------------------------------------------------------------------------------------- inference

def box_iou(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None] - inter, 1e-9)


def fuse_boxes(views, iou=0.45):
    """Weighted box fusion of several (boxes, scores) views of one image.

    Boxes are visited by descending score; a box joins the first-best fused box it overlaps with IoU >= iou, whose
    coordinates become the score-weighted mean of its members. Fused score = sum of member scores / number of views,
    capped at the best member score."""
    boxes = np.concatenate([b for b, _ in views]); scores = np.concatenate([s for _, s in views])
    order = np.argsort(-scores)
    fused, members_b, members_s = [], [], []
    for box, score in zip(boxes[order], scores[order]):
        if fused:
            overlaps = box_iou(box[None], np.asarray(fused))[0]
            j = int(np.argmax(overlaps))
            if overlaps[j] >= iou:
                members_b[j].append(box); members_s[j].append(score)
                w = np.asarray(members_s[j])
                fused[j] = (np.asarray(members_b[j]) * w[:, None]).sum(0) / w.sum()
                continue
        fused.append(box.copy()); members_b.append([box]); members_s.append([score])
    if not fused:
        return np.zeros((0, 4)), np.zeros(0)
    out = np.minimum(np.array([sum(m) / len(views) for m in members_s]), np.array([max(m) for m in members_s]))
    order = np.argsort(-out)
    return np.asarray(fused)[order], out[order]


def detect(model, image, min_score=0.01):
    """Fused boxes (xyxy, native pixels) and scores of one grayscale image over the eight views."""
    bgr = np.repeat(image[:, :, None], 3, axis=2)
    views = []
    for size in SIZES:
        for lr, ud in FLIPS.values():
            view = np.ascontiguousarray(bgr[::-1 if ud else 1, ::-1 if lr else 1])
            result = model.predict(view, imgsz=size, device=0, conf=min_score, iou=0.7, max_det=300, quantize=16, verbose=False)[0]
            boxes = result.boxes.xyxy.float().cpu().numpy().astype(float)
            scores = result.boxes.conf.float().cpu().numpy().astype(float)
            if lr:
                boxes[:, [0, 2]] = SIZE - boxes[:, [2, 0]]
            if ud:
                boxes[:, [1, 3]] = SIZE - boxes[:, [3, 1]]
            keep = scores >= min_score
            views.append((boxes[keep], scores[keep]))
    if sum(len(s) for _, s in views) == 0:
        return np.zeros((0, 4)), np.zeros(0)
    return fuse_boxes(views)


def propose(fold, splits=("heldout", "test")):
    """Boxes of detector `fold` on its held-out training images and/or the test images (JSON lines, one image per line)."""
    from ultralytics import YOLO
    model = YOLO(str(detector_path(fold)))
    for split in splits:
        stems, test = (test_stems(), True) if split == "test" else (fold_stems(fold), False)
        path = proposals_path(fold, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as out:
            for stem in stems:
                boxes, scores = detect(model, load_image(stem, test))
                out.write(json.dumps({"image_id": stem, "boxes_xyxy": np.round(boxes, 2).tolist(),
                                      "scores": np.round(scores, 5).tolist()}) + "\n")
        print(f"fold {fold} {split}: {len(stems)} images -> {path}", flush=True)


def read_proposals(path):
    """stem -> (boxes[n, 4], scores[n])"""
    out = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        out[row["image_id"]] = (np.asarray(row["boxes_xyxy"], float).reshape(-1, 4), np.asarray(row["scores"], float))
    return out
