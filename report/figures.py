#!/usr/bin/env python3
"""Figures and validation numbers of the report, from the outputs of reproduce.sh (run from the repository root).

  python report/figures.py   ->  report/figures/{pipeline,examples,density,agreement}.png, pipeline.pdf, numbers.json, numbers.tex"""
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from pycocotools import mask as cocomask

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run  # noqa: E402
from filaments import data, detector, instances, metric, segmenter  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231"]
OURS, ANNOTATORS = "#2a78d6", "#eb6834"   # agreement figure: our predictions, another annotator
YELLOW, PPI, DEPTH = "#f2d21b", 600, (0.42, 0.2)       # pipeline: zoom window, raster resolution, perspective
SANS = "Linux Biolinum O"                                  # the sans font of the acmart class
for font in Path("/usr/share/texlive/texmf-dist/fonts/opentype/public/libertine").glob("LinBiolinum_*.otf"):
    font_manager.fontManager.addfont(str(font))
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7})


def show(ax, image, window, title=None):
    x0, y0, x1, y1 = window
    ax.imshow(image[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255)
    ax.set_xlim(-0.5, x1 - x0 - 0.5); ax.set_ylim(y1 - y0 - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)


def outline(ax, rles, color, lw, window):
    for rle in rles:
        m, mx, my = data.decode_window(rle)
        for c in cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
            c = c[:, 0] + [mx - window[0], my - window[1]]
            ax.plot(*np.vstack([c, c[:1]]).T, color=color, lw=lw)


def window_around(rle, half):
    m, x0, y0 = data.decode_window(rle)
    cx, cy = x0 + m.shape[1] // 2, y0 + m.shape[0] // 2
    cx, cy = min(max(cx, half), data.SIZE - half), min(max(cy, half), data.SIZE - half)
    return (cx - half, cy - half, cx + half, cy + half)


def picture(image, window, inches, boxes=(), outlines=(), cmap="gray", vmax=255):
    """RGB picture of image[window] at PPI for a side of `inches`, with boxes [(xyxy, colour, pt)] and mask outlines
    [(rles, colour, pt)] drawn into the pixels, so that it can be shown in perspective."""
    x0, y0, x1, y1 = window
    scale = inches * PPI / max(x1 - x0, y1 - y0)
    pixels = cv2.resize(image[y0:y1, x0:x1].astype(np.float32), (round((x1 - x0) * scale), round((y1 - y0) * scale)),
                        interpolation=cv2.INTER_AREA)
    out = np.ascontiguousarray((plt.get_cmap(cmap)(np.clip(pixels / vmax, 0, 1))[..., :3] * 255).astype(np.uint8))
    width = lambda pt: max(1, round(pt * PPI / 72))                                       # noqa: E731
    color = lambda c: tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))                         # noqa: E731
    for b, c, pt in boxes:
        p = np.round((np.asarray(b, float) - [x0, y0, x0, y0]) * scale * 16).astype(int).tolist()
        cv2.rectangle(out, p[:2], p[2:], color(c), width(pt), cv2.LINE_AA, shift=4)
    for rles, c, pt in outlines:
        for rle in rles:
            m, mx, my = data.decode_window(rle)
            contours = [np.round((k[:, 0] + [mx - x0, my - y0]) * scale * 16).astype(np.int32)
                        for k in cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]]
            cv2.polylines(out, contours, True, color(c), width(pt), cv2.LINE_AA, shift=4)
    return out


def card(ax, pic, x, y, s, alpha=1.0, zorder=3):
    """A picture in perspective, standing in the depth plane: front edge at x from y to y + s."""
    t = Affine2D(np.array([[DEPTH[0] * s, 0, x], [DEPTH[1] * s, s, y], [0, 0, 1.0]])) + ax.transData
    ax.imshow(pic, extent=(0, 1, 0, 1), transform=t, alpha=alpha, zorder=zorder)
    corners = [(x, y), (x + DEPTH[0] * s, y + DEPTH[1] * s), (x + DEPTH[0] * s, y + DEPTH[1] * s + s), (x, y + s)]
    ax.add_patch(plt.Polygon(corners, fill=False, ec="0.25", lw=0.5, alpha=alpha, zorder=zorder + 0.05))


def slab(ax, x, y, w, h, fill, edge, zorder):
    """A feature map w thick and h high and deep: front, top and side faces, the latter two shaded."""
    dx, dy = DEPTH[0] * h, DEPTH[1] * h
    r, g, b = matplotlib.colors.to_rgb(fill)
    for pts, f in [([(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)], 0.8),
                   ([(x, y + h), (x + w, y + h), (x + w + dx, y + h + dy), (x + dx, y + h + dy)], 0.92),
                   ([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], 1.0)]:
        ax.add_patch(plt.Polygon(pts, fc=(r * f, g * f, b * f), ec=edge, lw=0.45, joinstyle="round", zorder=zorder))


def arrow(ax, x, y, length, text=None):
    ax.add_patch(FancyArrowPatch((x, y), (x + length, y), arrowstyle="-|>,head_length=3.0,head_width=1.7", color="0.2", lw=0.8,
                                 shrinkA=0, shrinkB=0, zorder=6))
    if text:
        ax.text(x + length / 2, y + 0.04, text, ha="center", va="bottom", fontsize=6.5, family=SANS, color="0.35", linespacing=1.0)


def pipeline_figure(stem, selection):
    """The pipeline on one test image: pictures as cards in perspective, the networks as stacks of feature maps."""
    image = data.load_image(stem, test=True)
    full = (0, 0, data.SIZE, data.SIZE)
    best = max(selection[stem], key=lambda x: x[1])[0]
    zoom = window_around(best, 300)
    model = segmenter.load("final")
    boxes, scores = detector.read_proposals(detector.proposals_path(0, "test"))[stem]
    probs, crops, kept, geometry = segmenter.predict(model, image, boxes, scores)
    order = np.argsort(-scores, kind="stable")[:len(crops)]
    bx, by, bw, bh = cocomask.toBbox(best)
    j = int(np.argmax(detector.box_iou([bx, by, bx + bw, by + bh], boxes[order])[0]))   # the crop of that filament
    x0, y0, x1, y1 = crop = tuple(int(v) for v in crops[j])
    b = np.round(boxes[order[j]] - [x0, y0, x0, y0]).astype(int)
    prompt = np.zeros((y1 - y0, x1 - x0), np.uint8)
    prompt[max(b[1], 0):b[3], max(b[0], 0):b[2]] = 255
    proposals = [detector.read_proposals(detector.proposals_path(k, "test"))[stem] for k in range(4)]
    candidates = [segmenter.read_candidates(run.masks_path("final", "test", k))[stem] for k in range(4)]

    W, H, S, y = 7.1, 1.42, 0.98, 0.2                   # inches: figure, side of a card, lower edge of the cards
    A, B = DEPTH
    mid = y + (S + B * S) / 2
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
    labels, x = [], 0.06

    for k, flip in enumerate([-1, 0, 1, None]):                                          # the eight views as a deck
        view = image if flip is None else cv2.flip(image, flip)
        card(ax, picture(view, full, S, boxes=[(zoom, YELLOW, 0.8)] if flip is None else ()), x + 0.07 * k, y, S,
             alpha=(0.35, 0.5, 0.7, 1)[k], zorder=3 + 0.1 * k)
    labels.append((x + (0.21 + A * S) / 2, "image, 8 views")); x += 0.21 + A * S + 0.08
    arrow(ax, x, mid, 0.16); x += 0.2
    start = x
    for i, (w, h) in enumerate([(0.05, 0.78), (0.06, 0.62), (0.07, 0.48), (0.08, 0.36), (0.09, 0.26)]):
        slab(ax, x, mid - (h + B * h) / 2, w, h, "#bcd3f2", "#4a6fa5", zorder=4 + 0.1 * i)
        x += w + 0.075
    x += A * 0.26
    labels.append(((start + x) / 2, "4 × YOLO26s"))
    arrow(ax, x + 0.02, mid, 0.18, "WBF"); x += 0.24
    nested = [(b, COLORS[k], 0.45 + 0.35 * k) for k in reversed(range(4)) for b in proposals[k][0][proposals[k][1] >= 0.2]]
    card(ax, picture(image, zoom, S, boxes=nested + [(crop, YELLOW, 0.7)]), x, y, S)
    labels.append((x + A * S / 2, "boxes")); x += A * S + 0.08
    arrow(ax, x, mid, 0.16); x += 0.2
    card(ax, picture(prompt, (0, 0, x1 - x0, y1 - y0), S), x, y, S)
    card(ax, picture(image, crop, S, boxes=[(boxes[order[j]], YELLOW, 1.0)]), x + 0.24, y, S, zorder=3.2)
    labels.append((x + (0.24 + A * S) / 2, "crop + box prompt")); x += 0.24 + A * S + 0.08
    arrow(ax, x, mid, 0.16); x += 0.2
    start, tops = x, []
    sizes = [(0.04, 0.8), (0.05, 0.6), (0.06, 0.44), (0.07, 0.32), (0.08, 0.24), (0.07, 0.32), (0.06, 0.44), (0.05, 0.6), (0.04, 0.8)]
    for i, (w, h) in enumerate(sizes):                                                   # encoder, bottleneck, decoder
        bottom = mid - (h + B * h) / 2
        slab(ax, x, bottom, w, h, "#f6c99c" if i < 4 else "#f3a76a" if i == 4 else "#fbe0c4", "#b36a24", zorder=4 + 0.1 * i)
        tops.append((x + w / 2 + A * h / 2, bottom + h + B * h + 0.01))
        x += w + 0.06
    x += A * 0.8 - 0.06
    for i in range(4):                                                                   # skip connections
        ax.add_patch(FancyArrowPatch(tops[i], tops[8 - i], arrowstyle="-|>,head_length=1.8,head_width=1.0", color="#b36a24", lw=0.5,
                                     connectionstyle=f"arc3,rad=-{0.28 - 0.03 * i}", shrinkA=0, shrinkB=0, alpha=0.85, zorder=8))
    labels.append(((start + x) / 2, "U-Net (ResNet18)"))
    arrow(ax, x + 0.02, mid, 0.16); x += 0.22
    card(ax, picture(segmenter.restore(probs[j], geometry[j]), (0, 0, x1 - x0, y1 - y0), S, cmap="magma", vmax=1), x, y, S)
    labels.append((x + A * S / 2, "probability")); x += A * S + 0.08
    arrow(ax, x, mid, 0.16); x += 0.2
    nested = [([r for score, r in candidates[k] if score >= 0.1], COLORS[k], 0.4 + 0.4 * k) for k in reversed(range(4))]
    card(ax, picture(image, zoom, S, outlines=nested), x, y, S)
    labels.append((x + A * S / 2, "candidates")); x += A * S + 0.08
    arrow(ax, x, mid, 0.24, "merge\nselect"); x += 0.28
    card(ax, picture(image, full, S, outlines=[([r for r, _ in selection[stem]], "#00e5ff", 0.9)]), x, y, S)
    labels.append((x + A * S / 2, "instances"))
    for lx, text in labels:
        ax.text(lx, y - 0.06, text, ha="center", va="top", fontsize=7.5, family=SANS, color="0.12")
    fig.savefig(OUT / "pipeline.pdf", dpi=600, metadata={"CreationDate": None})
    fig.savefig(OUT / "pipeline.png", dpi=300)


def examples_figure(stems, selection):
    recs = data.records()
    fig, axes = plt.subplots(2, len(stems), figsize=(7.1, 3.65))
    for col, stem in enumerate(stems):
        image = data.load_image(stem)
        biggest = max((rle for r in recs[stem] for rle in r["rles"]), key=lambda r: int(cocomask.area(r)))
        window = window_around(biggest, 256)
        show(axes[0, col], image, window, f"{stem[:8]}, {len(recs[stem])} annotators")
        for r, color in zip(recs[stem], COLORS):
            outline(axes[0, col], r["rles"], color, 0.6, window)
        show(axes[1, col], image, window)
        outline(axes[1, col], [r for r, _ in selection[stem]], "cyan", 0.6, window)
    axes[0, 0].set_ylabel("annotators"); axes[1, 0].set_ylabel("prediction")
    fig.subplots_adjust(left=0.025, right=0.995, top=0.95, bottom=0.005, wspace=0.03, hspace=0.03)
    fig.savefig(OUT / "examples.png", dpi=300)


def overlaps(pairs):
    """Statistics of the organisers' self-evaluation notebook over (truth RLEs, predicted RLEs) pairs, one per record:
    IoU of every overlapping pair, predictions overlapping each true filament (1:n) and true filaments overlapping each
    prediction (n:1), and the PQ counts."""
    ious, per_truth, per_pred, counts = [], [], [], np.zeros(4)
    for truth, pred in pairs:
        if not truth or not pred:
            per_truth += [0] * len(truth); per_pred += [0] * len(pred); counts += [0, 0, len(pred), len(truth)]
            continue
        iou = np.asarray(cocomask.iou(truth, pred, [0] * len(pred)))
        ious += iou[iou > 0].tolist()
        per_truth += (iou > 0).sum(1).tolist(); per_pred += (iou > 0).sum(0).tolist()
        hit = iou > 0.5
        counts += [iou[hit].sum(), hit.sum(), (hit.sum(0) == 0).sum(), (hit.sum(1) == 0).sum()]
    return np.asarray(ious), Counter(per_truth), Counter(per_pred), metric.panoptic_quality(counts)


def relation_shares(counter):
    total = sum(counter.values())
    return [counter[0] / total, counter[1] / total, sum(v for k, v in counter.items() if k >= 2) / total]


def matched(truth, pred):
    """Which true filaments have a prediction with IoU > 0.5."""
    if not truth or not pred:
        return np.zeros(len(truth), bool)
    return (np.asarray(cocomask.iou(truth, pred, [0] * len(pred))) > 0.5).any(1)


def agreement(stems, selection):
    """Agreement with the annotators on the validation fold: our predictions against each record, one annotator against
    another on the images with several records, recall by annotator consensus and by filament size, and how many of our
    false positives another annotator drew."""
    recs = data.records()
    pred = {s: [r for r, _ in selection[s]] for s in stems}
    multi = [s for s in stems if len(recs[s]) > 1]
    ours = overlaps([(r["rles"], pred[s]) for s in stems for r in recs[s]])
    ours_multi = overlaps([(r["rles"], pred[s]) for s in multi for r in recs[s]])
    between = overlaps([(a["rles"], b["rles"]) for s in multi for a in recs[s] for b in recs[s] if a is not b])

    consensus, sizes, false_pos = Counter(), [], [0, 0]
    for s in stems:
        for r in recs[s]:
            found = matched(r["rles"], pred[s])
            sizes += zip(cocomask.area(r["rles"]).tolist() if r["rles"] else [], found)
            others = [o["rles"] for o in recs[s] if o is not r]
            if len(recs[s]) == 3:              # how many of the three annotators drew each filament
                for g, hit in zip(r["rles"], found):
                    k = 1 + sum(bool(matched([g], o)[0]) for o in others)
                    consensus[k, "n"] += 1; consensus[k, "hit"] += int(hit)
            if len(recs[s]) > 1 and pred[s]:   # our false positives against this record that another annotator drew
                unmatched = [p for p, hit in zip(pred[s], matched(pred[s], r["rles"])) if not hit]
                false_pos[0] += len(unmatched)
                false_pos[1] += sum(any(matched([p], o)[0] for o in others) for p in unmatched)
    area, found = np.array([a for a, _ in sizes]), np.array([f for _, f in sizes])
    bins = [(0, 1000), (1000, 2500), (2500, 6000), (6000, np.inf)]

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 1.65))
    series = [(ours_multi, OURS), (between, ANNOTATORS)]          # the same images (two or three records) for both
    edges = np.linspace(0, 1, 26)
    for ax, (name, transform) in zip(axes[:2], [("IoU", lambda x: x), ("Dice", lambda x: 2 * x / (1 + x))]):
        means = []
        for (ious, *_), color in series:
            values = transform(ious)
            ax.hist(values, bins=edges, weights=np.full(len(values), 1 / len(values)), histtype="step", color=color, lw=1.2)
            means.append(values.mean())
        ax.text(0.02, 0.97, f"mean {means[0]:.2f} vs {means[1]:.2f}", transform=ax.transAxes, va="top", color="0.3")
        ax.set_xlabel(f"{name} of overlapping pairs"); ax.set_ylabel("share of pairs")
    ax = axes[2]
    x = np.array([0, 1, 2, 3.8, 4.8, 5.8])
    for offset, (result, color) in zip((-0.19, 0.19), series):
        ax.bar(x + offset, relation_shares(result[1]) + relation_shares(result[2]), width=0.36, color=color,
               edgecolor="white", lw=0.5)
    ax.set_xticks(x, ["1:0", "1:1", "1:2+", "0:1", "1:1", "2+:1"]); ax.set_ylabel("share")
    ax.text(1, -0.3, "per true filament", ha="center", transform=ax.get_xaxis_transform(), color="0.3")
    ax.text(4.8, -0.3, "per prediction", ha="center", transform=ax.get_xaxis_transform(), color="0.3")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (OURS, ANNOTATORS)]
    fig.legend(handles, ["ours vs annotator", "annotator vs another annotator"], loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.2); fig.savefig(OUT / "agreement.png", dpi=300)

    return {"ours": dict(ours[3], mean_iou=float(ours[0].mean()), mean_dice=float((2 * ours[0] / (1 + ours[0])).mean()),
                         truth=relation_shares(ours[1]), pred=relation_shares(ours[2])),
            "ours_multi": dict(ours_multi[3], mean_iou=float(ours_multi[0].mean()), truth=relation_shares(ours_multi[1]), pred=relation_shares(ours_multi[2])),
            "between": dict(between[3], mean_iou=float(between[0].mean()),
                            mean_dice=float((2 * between[0] / (1 + between[0])).mean()),
                            truth=relation_shares(between[1]), pred=relation_shares(between[2])),
            "consensus": {k: {"filaments": consensus[k, "n"], "recall": consensus[k, "hit"] / consensus[k, "n"]} for k in (1, 2, 3)},
            "size": [{"low": lo, "high": None if np.isinf(hi) else hi, "filaments": int(((area >= lo) & (area < hi)).sum()),
                      "recall": float(found[(area >= lo) & (area < hi)].mean())} for lo, hi in bins],
            "false_positives": {"multi": false_pos[0], "drawn_by_another": false_pos[1]}, "multi_images": len(multi)}


def write_macros(numbers):
    """The validation numbers as LaTeX macros for report.tex."""
    v, r = numbers["validation"], numbers["by_records"]
    macros = {"valPQ": f"{v['PQ']:.3f}", "valSQ": f"{v['SQ']:.3f}", "valRQ": f"{v['RQ']:.3f}", "valLo": f"{v['low']:.3f}",
              "valHi": f"{v['high']:.3f}", "valTP": f"{v['TP']:,}", "valFP": f"{v['FP']:,}", "valFN": f"{v['FN']:,}",
              "valPQone": f"{r[1]['PQ']:.3f}", "valPQtwo": f"{r[2]['PQ']:.3f}", "valPQthree": f"{r[3]['PQ']:.3f}",
              "testInstances": f"{numbers['test']['instances']:,}"}
    a = numbers["agreement"]
    pct = lambda x: f"{100 * x:.0f}"                                                     # noqa: E731
    macros.update({"agIoU": f"{a['ours']['mean_iou']:.2f}", "agDice": f"{a['ours']['mean_dice']:.2f}",
                   "agTruthOne": pct(a["ours"]["truth"][1]), "agTruthNone": pct(a["ours"]["truth"][0]),
                   "agTruthMany": f"{100 * a['ours']['truth'][2]:.1f}", "agPredOne": pct(a["ours"]["pred"][1]),
                   "agPredNone": pct(a["ours"]["pred"][0]), "agPredMany": f"{100 * a['ours']['pred'][2]:.1f}",
                   "multiPQ": f"{a['ours_multi']['PQ']:.3f}", "multiSQ": f"{a['ours_multi']['SQ']:.3f}",
                   "multiIoU": f"{a['ours_multi']['mean_iou']:.2f}", "annPQ": f"{a['between']['PQ']:.3f}",
                   "annSQ": f"{a['between']['SQ']:.3f}", "annIoU": f"{a['between']['mean_iou']:.2f}",
                   "annDice": f"{a['between']['mean_dice']:.2f}", "annTruthOne": pct(a["between"]["truth"][1]),
                   "fpMulti": f"{a['false_positives']['multi']:,}", "multiImages": str(a["multi_images"]),
                   "fpOther": pct(a["false_positives"]["drawn_by_another"] / a["false_positives"]["multi"])})
    for k, word in zip((1, 2, 3), ("One", "Two", "Three")):
        macros[f"cons{word}N"] = f"{a['consensus'][k]['filaments']:,}"
        macros[f"cons{word}R"] = f"{a['consensus'][k]['recall']:.2f}"
    for i, word in enumerate(("A", "B", "C", "D")):
        macros[f"size{word}N"] = f"{a['size'][i]['filaments']:,}"
        macros[f"size{word}R"] = f"{a['size'][i]['recall']:.2f}"
    (OUT / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))


def main():
    OUT.mkdir(exist_ok=True)
    recs = data.records()
    val_stems = data.fold_stems(0)
    val_objects = run.objects("validation", "heldout", [0], val_stems)
    rng = np.random.default_rng(0)
    draws = [rng.integers(0, len(val_stems), len(val_stems)) for _ in range(2000)]
    curve, numbers = [], {}
    for density in np.round(np.arange(4.2, 8.21, 0.4), 1):
        selection, cutoff = instances.select_at_density(val_objects, density)
        rows = np.array([metric.image_counts(recs[s], [r for r, _ in selection[s]]) for s in val_stems])
        boot = [metric.panoptic_quality(rows[d].sum(0))["PQ"] for d in draws]
        q = metric.panoptic_quality(rows.sum(0))
        curve.append((density, q["PQ"], np.percentile(boot, 2.5), np.percentile(boot, 97.5)))
        if abs(density - instances.DENSITY) < 1e-6:
            numbers["validation"] = dict(q, cutoff=cutoff, low=float(np.percentile(boot, 2.5)), high=float(np.percentile(boot, 97.5)))
            numbers["by_records"] = {n: metric.panoptic_quality(rows[[len(recs[s]) == n for s in val_stems]].sum(0))
                                     for n in (1, 2, 3)}
            val_selection = selection
    numbers["density_curve"] = [dict(density=float(d), PQ=float(q), low=float(a), high=float(b)) for d, q, a, b in curve]
    d, pq, lo, hi = map(np.array, zip(*curve))
    fig, ax = plt.subplots(figsize=(3.3, 1.7))
    ax.fill_between(d, lo, hi, color="C0", alpha=0.2, lw=0, label="95% bootstrap interval")
    ax.plot(d, pq, "o-", color="C0", ms=3, label="fold 0")
    ax.axvline(instances.DENSITY, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("instances per image"); ax.set_ylabel("PQ"); ax.legend(frameon=False, loc="lower center")
    fig.tight_layout(); fig.savefig(OUT / "density.png", dpi=300)

    test_selection, cutoff = instances.select_at_density(run.objects("final", "test", range(4), data.test_stems()))
    numbers["test"] = {"instances": sum(map(len, test_selection.values())), "cutoff": cutoff}
    pipeline_figure(max(test_selection, key=lambda s: len(test_selection[s])), test_selection)
    examples_figure([s for s in val_stems if len(recs[s]) == 3][1:5], val_selection)
    numbers["agreement"] = agreement(val_stems, val_selection)
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float) + "\n")
    write_macros(numbers)
    print(json.dumps(numbers, indent=1, default=float))


if __name__ == "__main__":
    main()
