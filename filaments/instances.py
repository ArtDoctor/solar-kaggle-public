"""From candidate masks to the final instances: detector ensemble, pixel-disjoint selection, submission.

1. Candidates (segmenter.candidates): each box's probability map is thresholded at 0.40; the largest connected piece
   is the candidate mask (at least 150 px), scored by the detector confidence of its box.
2. Ensemble: candidates of the four detectors that overlap with mask IoU > 0.5 are one object. Its score is the mean
   over the four detectors (a detector that missed it adds 0); its mask is that of its highest-scoring member.
3. Selection: objects are taken in score order down to a cutoff; each loses the pixels already taken, keeps only its
   largest remaining piece and is dropped if less than 150 px or 40% of it remains. Kept pieces closer than 8 px are
   merged into one instance (a filament broken into fragments). The cutoff is set to give 6.2 instances per image.
"""
import csv
from multiprocessing import Pool

import cv2
import numpy as np
from pycocotools import mask as cocomask

from .data import SIZE, decode_window, encode, largest_component

MIN_AREA, MERGE_PX, DENSITY = 150, 8, 6.2


def ensemble(per_detector, iou=0.5):
    """Merge the candidate lists [(score, rle)] of several detectors of one image into objects: (rles, scores)."""
    ranked = sorted(((score, d, rle) for d, cands in enumerate(per_detector) for score, rle in cands), key=lambda x: -x[0])
    objects = []
    for score, d, rle in ranked:
        if objects:
            overlaps = cocomask.iou([rle], [o["rle"] for o in objects], [0] * len(objects))[0]
            j = int(np.argmax(overlaps))
            if overlaps[j] > iou:
                if d not in objects[j]["detectors"]:     # each detector votes once per object
                    objects[j]["detectors"].add(d); objects[j]["scores"].append(score)
                continue
        objects.append({"rle": rle, "detectors": {d}, "scores": [score]})
    return [o["rle"] for o in objects], np.array([sum(o["scores"]) / len(per_detector) for o in objects])


def select(rles, scores, cutoff, min_area=MIN_AREA, duplicate_iou=0.8, min_keep=0.4, merge_px=MERGE_PX):
    """Greedy pixel-disjoint selection of one image; returns [(rle, score)]."""
    taken = np.zeros((SIZE, SIZE), bool)
    originals, kept = [], []
    for i in np.argsort(-np.asarray(scores)):
        if scores[i] < cutoff:
            break
        rle = rles[i]
        area = int(cocomask.area(rle))
        if area < min_area or (originals and (cocomask.iou([rle], originals, [0] * len(originals)) >= duplicate_iou).any()):
            continue
        mask, x0, y0 = decode_window(rle)
        window = taken[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]]
        rest = mask & ~window
        if rest.sum() < min_area:
            continue
        rest = largest_component(rest)
        if rest is None or rest.sum() < min_area or rest.sum() / area < min_keep:
            continue
        window |= rest
        originals.append(rle)
        kept.append((rest, x0, y0, float(scores[i])))
    return merge_close(kept, merge_px)


def merge_close(kept, merge_px):
    """Union of kept pieces whose pixel gap is at most merge_px; a merged instance keeps its best score."""
    parent = list(range(len(kept)))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    boxes = [(x0, y0, x0 + m.shape[1], y0 + m.shape[0]) for m, x0, y0, _ in kept]
    for a in range(len(kept)):
        for b in range(a + 1, len(kept)):
            A, B = boxes[a], boxes[b]
            gap = np.hypot(max(0, max(A[0], B[0]) - min(A[2], B[2])), max(0, max(A[1], B[1]) - min(A[3], B[3])))
            if merge_px <= 0 or gap > merge_px:
                continue
            ux0, uy0, ux1, uy1 = min(A[0], B[0]), min(A[1], B[1]), max(A[2], B[2]), max(A[3], B[3])
            canvas = np.ones((uy1 - uy0, ux1 - ux0), np.uint8)
            canvas[A[1] - uy0:A[3] - uy0, A[0] - ux0:A[2] - ux0][kept[a][0]] = 0
            distance = cv2.distanceTransform(canvas, cv2.DIST_L2, 3)
            if distance[B[1] - uy0:B[3] - uy0, B[0] - ux0:B[2] - ux0][kept[b][0]].min() <= merge_px:
                parent[root(a)] = root(b)
    groups = {}
    for a in range(len(kept)):
        groups.setdefault(root(a), []).append(kept[a])
    out = []
    for group in groups.values():
        pieces = [encode(m, x0, y0) for m, x0, y0, _ in group]
        out.append((cocomask.merge(pieces) if len(pieces) > 1 else pieces[0], max(s for *_, s in group)))
    return out


def _select(task):
    stem, rles, scores, cutoff = task
    return stem, select(rles, scores, cutoff)


def select_at_density(objects, density=DENSITY, workers=8):
    """Selection for every image with the cutoff that gives `density` instances per image (bisection)."""
    def run(cutoff):
        with Pool(workers) as pool:
            return dict(pool.map(_select, [(s, r, sc, cutoff) for s, (r, sc) in objects.items()], chunksize=4))
    lo, hi = 0.05, 0.6
    for _ in range(12):
        mid = (lo + hi) / 2
        if sum(map(len, run(mid).values())) / len(objects) > density:
            lo = mid
        else:
            hi = mid
    cutoff = (lo + hi) / 2
    return run(cutoff), cutoff


def write_submission(path, selections):
    """CSV with one row per instance: <image stem>_<rank>, COCO RLE counts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filament_id", "segmentation_rle"])
        for stem in sorted(selections):
            for i, (rle, _) in enumerate(sorted(selections[stem], key=lambda k: -k[1]), 1):
                writer.writerow([f"{stem}_{i}", rle["counts"].decode("ascii")])
