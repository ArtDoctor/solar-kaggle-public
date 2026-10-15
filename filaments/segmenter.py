"""Box-prompted mask model: one filament mask per detector box.

For every box, a square crop with 30% padding (at least 32 px) is cut from the native image and letterboxed to 512 px.
The network sees two channels, the crop and a binary box prompt, and returns the probability that each pixel belongs to
the filament in the box. Architecture: U-Net with a ResNet18 encoder (ImageNet weights).

Training crops come from the detectors' out-of-fold boxes, so the prompts look like the ones met at test time: each
annotated filament is prompted with its best-matching detector box (box IoU >= 0.3, whole filament inside the crop), or
with its own box if no detector box qualifies; the two highest-scoring boxes that touch no annotation of any annotator
become empty-target crops. Every epoch takes one annotator record per image, so the model learns each annotator's
outlines rather than their union.
"""
import json
import random
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as cocomask

from .data import SIZE, WORK, encode, folds, largest_component, load_image, records, train_stems
from .detector import box_iou, proposals_path, read_proposals

PADDING, MIN_PADDING, INPUT = 0.3, 32, 512
MAX_BOXES = 40                                   # highest-scoring boxes per image and detector that get a mask
THRESHOLD, MIN_AREA = 0.40, 150                  # probability threshold and smallest candidate mask (px)
TRAIN = dict(epochs=24, batch=16, lr=3e-4, weight_decay=1e-4, jitter=0.25, seed=104)


def segmenter_path(name):
    return WORK / "weights" / f"segmenter_{name}.pt"


def build_model(pretrained=True):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name="resnet18", encoder_weights="imagenet" if pretrained else None, in_channels=2, classes=1)


# ------------------------------------------------------------------------------------------------------ crop geometry

def square_crop(box):
    """Square window around a box with 30% padding (at least 32 px), clipped to the image: [x0, y0, x1, y1]."""
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0)
    half = side / 2 + max(MIN_PADDING, PADDING * side)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return [max(0, int(np.floor(cx - half))), max(0, int(np.floor(cy - half))),
            min(SIZE, int(np.ceil(cx + half))), min(SIZE, int(np.ceil(cy + half)))]


def letterbox(array, nearest=False):
    """Aspect-preserving resize to 512 px plus zero padding, and the geometry needed to undo it."""
    height, width = array.shape
    scale = min(INPUT / height, INPUT / width)
    rh, rw = min(INPUT, max(1, round(height * scale))), min(INPUT, max(1, round(width * scale)))
    resized = np.asarray(Image.fromarray(array).resize((rw, rh), Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR))
    top, left = (INPUT - rh) // 2, (INPUT - rw) // 2
    out = np.zeros((INPUT, INPUT), dtype=resized.dtype)
    out[top:top + rh, left:left + rw] = resized
    return out, (top, left, rh, rw, height, width)


def make_input(crop, box):
    """Network input (2 x 512 x 512): normalised crop and binary box prompt; box in crop coordinates."""
    prompt = np.zeros(crop.shape, np.float32)
    x0, y0 = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
    x1, y1 = min(crop.shape[1], int(np.ceil(box[2]))), min(crop.shape[0], int(np.ceil(box[3])))
    prompt[y0:y1, x0:x1] = 1
    grey, geometry = letterbox(crop)
    prompt, _ = letterbox(prompt, nearest=True)
    grey = (grey.astype(np.float32) / 255 - .449) / .226
    return np.stack([grey, prompt]).astype(np.float32), geometry


def restore(probability, geometry):
    """512 px probability map -> probability over the native crop."""
    top, left, rh, rw, height, width = geometry
    content = probability[top:top + rh, left:left + rw].astype(np.float32)
    return np.asarray(Image.fromarray(content).resize((width, height), Image.Resampling.BILINEAR))


# ----------------------------------------------------------------------------------------------------------- training

def _mask_box(mask):
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _image_crops(task):
    """Training crops of one image: (image crop, target crop, box in crop, record id, is_negative) per crop."""
    stem, boxes, scores = task
    image = load_image(stem)
    masks = {r["id"]: [cocomask.decode(rle).astype(bool) for rle in r["rles"]] for r in records()[stem]}
    anything = np.zeros((SIZE, SIZE), bool)
    for record_masks in masks.values():
        for m in record_masks:
            anything |= m
    plans = []                                   # (record id, mask or None, prompt box)
    for record_id, record_masks in masks.items():
        best = {}
        for index, box in enumerate(boxes):
            x0, y0, x1, y1 = square_crop(box)
            bx0, by0 = np.floor(box[:2]).astype(int); bx1, by1 = np.ceil(box[2:]).astype(int)
            matches = [(box_iou(box, _mask_box(m))[0, 0], a) for a, m in enumerate(record_masks) if m[by0:by1, bx0:bx1].any()]
            if not matches:
                continue
            quality, a = max(matches)
            if quality < 0.3 or record_masks[a][y0:y1, x0:x1].sum() != record_masks[a].sum():
                continue
            if a not in best or (quality, scores[index], -index) > best[a][0]:
                best[a] = ((quality, scores[index], -index), box)
        for a, m in enumerate(record_masks):
            plans.append((record_id, m, best[a][1] if a in best else _mask_box(m)))
    negatives = 0
    for box in boxes:
        if negatives == 2:
            break
        x0, y0, x1, y1 = square_crop(box)
        if not anything[y0:y1, x0:x1].any():
            plans.append((next(iter(masks)), None, box)); negatives += 1
    out = []
    for record_id, mask, box in plans:
        x0, y0, x1, y1 = square_crop(box)
        target = np.zeros((y1 - y0, x1 - x0), np.uint8) if mask is None else mask[y0:y1, x0:x1].astype(np.uint8)
        out.append({"stem": stem, "record": record_id, "negative": mask is None, "image": image[y0:y1, x0:x1].copy(),
                    "target": target, "box": np.asarray(box, float) - [x0, y0, x0, y0]})
    return out


class CropSet(torch.utils.data.Dataset):
    """Crops with box jitter, flips, 90-degree rotations and contrast / brightness changes."""
    def __init__(self, crops, jitter):
        self.crops, self.jitter = crops, jitter

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, index):
        c = self.crops[index]
        box = c["box"].copy()
        w, h = box[2] - box[0], box[3] - box[1]
        box += np.random.uniform(-self.jitter, self.jitter, 4) * np.array([w, h, w, h])
        box[0::2] = np.clip(box[0::2], 0, c["image"].shape[1])
        box[1::2] = np.clip(box[1::2], 0, c["image"].shape[0])
        inputs, _ = make_input(c["image"], box)
        target, _ = letterbox(c["target"], nearest=True)
        if np.random.random() < .5:
            inputs, target = inputs[:, :, ::-1], target[:, ::-1]
        if np.random.random() < .5:
            inputs, target = inputs[:, ::-1, :], target[::-1, :]
        k = int(np.random.randint(0, 4))
        inputs, target = np.rot90(inputs, k, axes=(1, 2)).copy(), np.rot90(target, k).copy()
        inputs[0] = inputs[0] * np.random.uniform(0.85, 1.15) + np.random.uniform(-.15, .15)
        return torch.from_numpy(inputs), torch.from_numpy(target[None].astype(np.float32))


class RecordSampler(torch.utils.data.Sampler):
    """Per epoch and image: the crops of one randomly chosen annotator record, plus the image's empty-target crops."""
    def __init__(self, crops, image_records, seed):
        self.records, self.seed, self.epoch = image_records, seed, 0
        self.positives, self.negatives = {}, {}
        for i, c in enumerate(crops):
            if c["negative"]:
                self.negatives.setdefault(c["stem"], []).append(i)
            else:
                self.positives.setdefault(c["stem"], {}).setdefault(c["record"], []).append(i)

    def indices(self):
        rng = random.Random(self.seed + self.epoch)
        out = []
        for stem in sorted(self.records):
            out += self.positives.get(stem, {}).get(rng.choice(self.records[stem]), []) + self.negatives.get(stem, [])
        rng.shuffle(out)
        return out

    def __iter__(self):
        return iter(self.indices())

    def __len__(self):
        return len(self.indices())


def bce_dice(logits, target):
    p = torch.sigmoid(logits.float())
    dice = (2 * (p * target).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return F.binary_cross_entropy_with_logits(logits.float(), target) + 1 - dice.mean()


def training_crops(stems):
    """Training crops of these images, prompted with the out-of-fold boxes of the four detectors."""
    boxes = {}
    for fold in range(4):
        boxes.update(read_proposals(proposals_path(fold, "heldout")))
    tasks = []
    for s in stems:
        b, sc = boxes[s]
        top = np.argsort(-sc, kind="stable")[:MAX_BOXES]
        b, sc = np.clip(b[top], 0, SIZE), sc[top]
        valid = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
        tasks.append((s, b[valid], sc[valid]))
    records()                                     # parse the annotations once, before the workers fork
    with Pool(6) as pool:
        return [c for chunk in pool.imap(_image_crops, tasks, chunksize=4) for c in chunk]


def train(name, exclude_fold=None, workers=5):
    """Train on every training image outside `exclude_fold`; the last epoch is kept."""
    cfg = TRAIN
    stems = [s for s in train_stems() if folds()[s] != exclude_fold]
    crops = training_crops(stems)
    print(f"{name}: {len(stems)} images, {len(crops)} crops ({sum(c['negative'] for c in crops)} empty)", flush=True)

    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda")
    sampler = RecordSampler(crops, {s: [r["id"] for r in records()[s]] for s in stems}, cfg["seed"])
    loader = torch.utils.data.DataLoader(CropSet(crops, cfg["jitter"]), batch_size=cfg["batch"], sampler=sampler,
                                         num_workers=workers, pin_memory=True, persistent_workers=True)
    for epoch in range(cfg["epochs"]):
        sampler.epoch = epoch
        model.train()
        total, n = 0.0, 0
        for inputs, target in loader:
            inputs, target = inputs.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda"):
                loss = bce_dice(model(inputs), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(inputs); n += len(inputs)
        scheduler.step()
        print(f"{name} epoch {epoch + 1}/{cfg['epochs']}: {n} crops, loss {total / n:.4f}", flush=True)
    segmenter_path(name).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}, "config": cfg, "exclude_fold": exclude_fold},
               segmenter_path(name))


# ---------------------------------------------------------------------------------------------------------- inference

def load(name):
    """A trained mask model on the GPU, in evaluation mode."""
    model = build_model(pretrained=False).cuda().eval()
    model.load_state_dict(torch.load(segmenter_path(name), map_location="cpu", weights_only=True)["model"])
    return model


@torch.inference_mode()
def predict(model, image, boxes, scores, batch=4):
    """Mask probabilities of the MAX_BOXES highest-scoring boxes of one image, averaged over four flips.

    Returns probs (n x 512 x 512, float16), crops (n x 4: x0 y0 x1 y1), scores (n) and geometry (n x 6, see letterbox)."""
    inputs, crops, kept, geometry = [], [], [], []
    for i in np.argsort(-scores, kind="stable")[:MAX_BOXES]:
        box = np.clip(boxes[i], 0, SIZE)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        x0, y0, x1, y1 = square_crop(box)
        x, g = make_input(image[y0:y1, x0:x1], box - [x0, y0, x0, y0])
        inputs.append(x); crops.append([x0, y0, x1, y1]); kept.append(scores[i]); geometry.append(g)
    probs = []
    for s in range(0, len(inputs), batch):
        x = torch.from_numpy(np.stack(inputs[s:s + batch])).cuda()
        p = torch.sigmoid(model(x)[:, 0]).float()
        for dims in ((-1,), (-2,), (-2, -1)):
            p += torch.flip(torch.sigmoid(model(torch.flip(x, dims))[:, 0]).float(), dims)
        probs.extend((p / 4).cpu().numpy())
    return (np.asarray(probs, np.float16).reshape(-1, INPUT, INPUT), np.asarray(crops, np.int32).reshape(-1, 4),
            np.asarray(kept, np.float32), np.asarray(geometry, np.int32).reshape(-1, 6))


def candidates(probs, crops, scores, geometry):
    """Candidate masks of one image: the largest piece of each box's probability >= 0.40, if at least 150 px.
    Returns [(score, full-canvas RLE)]."""
    out = []
    for prob, crop, score, g in zip(probs, crops, scores, geometry):
        mask = largest_component(restore(prob, g) >= THRESHOLD)
        if mask is not None and mask.sum() >= MIN_AREA:
            out.append((float(score), encode(mask, int(crop[0]), int(crop[1]))))
    return out


def segment(name, proposals, out_path, test):
    """Candidate masks for every image of a proposals file; JSON lines {"image_id", "scores", "rles" (COCO counts)}."""
    model = load(name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as out:
        for stem, (boxes, scores) in sorted(read_proposals(proposals).items()):
            cands = candidates(*predict(model, load_image(stem, test), boxes, scores))
            out.write(json.dumps({"image_id": stem, "scores": [c[0] for c in cands],
                                  "rles": [c[1]["counts"].decode("ascii") for c in cands]}) + "\n")
    print(f"{name}: masks for {proposals.name} -> {out_path}", flush=True)


def read_candidates(path):
    """stem -> [(score, RLE)] as written by segment()."""
    out = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        out[row["image_id"]] = [(score, {"size": [SIZE, SIZE], "counts": counts.encode("ascii")})
                                for score, counts in zip(row["scores"], row["rles"])]
    return out
