#!/usr/bin/env python3
"""Figures and validation numbers of the report, from the outputs of reproduce.sh (run from the repository root).

  python report/figures.py   ->  report/figures/{pipeline,examples,density}.png, numbers.json, numbers.tex"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from pycocotools import mask as cocomask

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run  # noqa: E402
from filaments import data, detector, instances, metric, segmenter  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231"]
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


def pipeline_figure(stem, selection):
    image = data.load_image(stem, test=True)
    full = (0, 0, data.SIZE, data.SIZE)
    best = max(selection[stem], key=lambda x: x[1])[0]
    zoom = window_around(best, 300)
    fig, axes = plt.subplots(1, 6, figsize=(7.1, 1.55))
    show(axes[0], image, full, "input image")
    axes[0].add_patch(plt.Rectangle(zoom[:2], 600, 600, fill=False, color="y", lw=0.6))
    show(axes[1], image, zoom, "boxes of 4 detectors")
    for k, color in enumerate(COLORS):
        boxes, scores = detector.read_proposals(detector.proposals_path(k, "test"))[stem]
        for b in boxes[scores >= 0.2]:
            axes[1].add_patch(plt.Rectangle((b[0] - zoom[0], b[1] - zoom[1]), b[2] - b[0], b[3] - b[1], fill=False, color=color, lw=0.6))
    model = segmenter.load("final")
    boxes, scores = detector.read_proposals(detector.proposals_path(0, "test"))[stem]
    probs, crops, kept, geometry = segmenter.predict(model, image, boxes, scores)
    order = np.argsort(-scores, kind="stable")[:len(crops)]
    bx, by, bw, bh = cocomask.toBbox(best)
    j = int(np.argmax(detector.box_iou([bx, by, bx + bw, by + bh], boxes[order])[0]))   # the crop of that filament
    x0, y0, x1, y1 = crops[j]
    b = boxes[order[j]] - [x0, y0, x0, y0]
    show(axes[2], image, (x0, y0, x1, y1), "crop + box prompt")
    axes[2].add_patch(plt.Rectangle(b[:2], b[2] - b[0], b[3] - b[1], fill=False, color="y", lw=0.8))
    axes[3].imshow(segmenter.restore(probs[j], geometry[j]), cmap="magma", vmin=0, vmax=1)
    axes[3].set_xticks([]); axes[3].set_yticks([]); axes[3].set_title("U-Net probability")
    show(axes[4], image, zoom, "ensemble + selection")
    outline(axes[4], [r for r, _ in selection[stem]], "cyan", 0.7, zoom)
    show(axes[5], image, full, "one mask per filament")
    outline(axes[5], [r for r, _ in selection[stem]], "cyan", 0.4, full)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.86, bottom=0.01, wspace=0.06)
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


def write_macros(numbers):
    """The validation numbers as LaTeX macros for report.tex."""
    v, r = numbers["validation"], numbers["by_records"]
    macros = {"valPQ": f"{v['PQ']:.3f}", "valSQ": f"{v['SQ']:.3f}", "valRQ": f"{v['RQ']:.3f}", "valLo": f"{v['low']:.3f}",
              "valHi": f"{v['high']:.3f}", "valTP": f"{v['TP']:,}", "valFP": f"{v['FP']:,}", "valFN": f"{v['FN']:,}",
              "valPQone": f"{r[1]['PQ']:.3f}", "valPQtwo": f"{r[2]['PQ']:.3f}", "valPQthree": f"{r[3]['PQ']:.3f}",
              "testInstances": f"{numbers['test']['instances']:,}"}
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
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    ax.fill_between(d, lo, hi, color="C0", alpha=0.2, lw=0, label="95% bootstrap interval")
    ax.plot(d, pq, "o-", color="C0", ms=3, label="fold 0")
    ax.axvline(instances.DENSITY, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("instances per image"); ax.set_ylabel("PQ"); ax.legend(frameon=False, loc="lower center")
    fig.tight_layout(); fig.savefig(OUT / "density.png", dpi=300)

    test_selection, cutoff = instances.select_at_density(run.objects("final", "test", range(4), data.test_stems()))
    numbers["test"] = {"instances": sum(map(len, test_selection.values())), "cutoff": cutoff}
    pipeline_figure(max(test_selection, key=lambda s: len(test_selection[s])), test_selection)
    examples_figure([s for s in val_stems if len(recs[s]) == 3][1:5], val_selection)
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float) + "\n")
    write_macros(numbers)
    print(json.dumps(numbers, indent=1, default=float))


if __name__ == "__main__":
    main()
