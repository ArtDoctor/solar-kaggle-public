#!/usr/bin/env python3
"""Command line for the whole pipeline; reproduce.sh runs the steps in order.

  python run.py tiles                                   detector training tiles for the four folds
  python run.py train-detector --fold K                 detector K, trained without fold K
  python run.py propose --fold K [--split S]            boxes of detector K on fold K (held out) and/or the test set
  python run.py train-segmenter --name NAME [--exclude-fold K]
  python run.py segment --name NAME --split heldout|test --fold K
  python run.py evaluate --name NAME                    PQ on fold 0: detector 0 + a mask model trained without fold 0
  python run.py submit --name NAME [--out FILE]         four detectors + mask model NAME on the test set -> CSV
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

from filaments import detector, instances, segmenter
from filaments.data import WORK, fold_stems, records, test_stems
from filaments.metric import image_counts, panoptic_quality


def masks_path(name, split, fold):
    return WORK / "masks" / f"{name}_{split}_fold{fold}.jsonl"


def _objects(task):
    stem, per_detector = task
    return stem, instances.ensemble(per_detector)


def objects(name, split, folds, stems):
    """stem -> (rles, scores) of the detector ensemble over `folds`."""
    cands = [segmenter.read_candidates(masks_path(name, split, k)) for k in folds]
    with Pool(8) as pool:
        return dict(pool.map(_objects, [(s, [c[s] for c in cands]) for s in stems], chunksize=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["tiles", "train-detector", "propose", "train-segmenter", "segment", "evaluate", "submit"])
    ap.add_argument("--fold", type=int)
    ap.add_argument("--name", default="final")
    ap.add_argument("--exclude-fold", type=int)
    ap.add_argument("--split", choices=["heldout", "test"])
    ap.add_argument("--density", type=float, default=instances.DENSITY)
    ap.add_argument("--out", default=str(WORK / "submission.csv"))
    a = ap.parse_args()

    if a.step == "tiles":
        detector.build_tiles()
    elif a.step == "train-detector":
        detector.train(a.fold)
    elif a.step == "propose":
        detector.propose(a.fold, [a.split] if a.split else ["heldout", "test"])
    elif a.step == "train-segmenter":
        segmenter.train(a.name, a.exclude_fold)
    elif a.step == "segment":
        segmenter.segment(a.name, detector.proposals_path(a.fold, a.split), masks_path(a.name, a.split, a.fold), a.split == "test")
    elif a.step == "evaluate":
        stems = fold_stems(0)
        selections, cutoff = instances.select_at_density(objects(a.name, "heldout", [0], stems), a.density)
        counts = sum(image_counts(records()[s], [rle for rle, _ in selections[s]]) for s in stems)
        n = sum(map(len, selections.values())) / len(stems)
        print(f"fold 0, {len(stems)} images, {n:.2f} instances per image (cutoff {cutoff:.4f}):",
              " ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in panoptic_quality(counts).items()))
    elif a.step == "submit":
        selections, cutoff = instances.select_at_density(objects(a.name, "test", range(4), test_stems()), a.density)
        instances.write_submission(Path(a.out), selections)
        n = sum(map(len, selections.values()))
        print(f"{n} instances on {len(selections)} images ({n / len(selections):.2f} per image, cutoff {cutoff:.4f}) -> {a.out}")


if __name__ == "__main__":
    main()
