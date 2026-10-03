#!/usr/bin/env python
"""Validate the identity scorer on real photographs.

This is the number worth believing in the README. The synthetic renderer exists to
exercise the pipeline, but a face embedding model trained on photographs cannot be trusted
on flat vector art — so the instrument gets checked where it was trained, against a real
dataset with real same-person/different-person labels.

Same-person vs different-person on the Olivetti face dataset (400 images, 40 identities),
which is held out by construction: the labels come with the data, and nothing about the
model saw them.

    python scripts/validate_scorer.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.services.identity import auroc, calibrate_threshold, cosine_matrix, get_scorer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    parser.add_argument("--data-home", default="data/sklearn")
    parser.add_argument(
        "--fail-below",
        type=float,
        default=0.95,
        help="exit non-zero if AUROC drops below this (default 0.95)",
    )
    args = parser.parse_args()

    try:
        from sklearn.datasets import fetch_olivetti_faces
    except ImportError:
        print("scikit-learn is required: pip install scikit-learn", file=sys.stderr)
        return 2

    started = time.time()
    dataset = fetch_olivetti_faces(data_home=args.data_home, download_if_missing=True)
    scorer = get_scorer()

    # Olivetti ships 64x64 grayscale portraits; widen to RGB so the model's normalisation
    # sees three channels like the photographs it was trained on.
    images = [(np.stack([frame] * 3, axis=-1) * 255).astype(np.uint8) for frame in dataset.images]
    embeddings, detected = scorer.embed_many(images)
    matrix = cosine_matrix(embeddings)
    labels = dataset.target
    n = len(labels)

    same_mask = labels[:, None] == labels[None, :]
    same = matrix[same_mask]
    different = matrix[~same_mask]

    # Drop self-comparisons, which are trivially 1.0 and would flatter every statistic.
    # `same` is the diagonal's entries in flattened order, so re-index the diagonal flags
    # through the same mask to find them.
    keep = ~np.eye(n, dtype=bool).ravel()[same_mask.ravel()]
    same = same[keep]

    positives = same.tolist()
    negatives = different.tolist()
    metrics = {
        "dataset": "Olivetti faces",
        "images": n,
        "identities": int(len(np.unique(labels))),
        "detection_rate": round(float(np.mean(detected)), 4),
        "same_person": {
            "mean": round(float(same.mean()), 4),
            "p05": round(float(np.percentile(same, 5)), 4),
            "pairs": int(same.size),
        },
        "different_person": {
            "mean": round(float(different.mean()), 4),
            "p95": round(float(np.percentile(different, 95)), 4),
            "pairs": int(different.size),
        },
        "auroc": round(auroc(positives, negatives), 4),
        **calibrate_threshold(positives, negatives),
        "elapsed_s": round(time.time() - started, 2),
    }

    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print("Identity scorer validation")
        print(f"  dataset          {metrics['dataset']} ({metrics['images']} images, "
              f"{metrics['identities']} identities)")
        print(f"  detection rate   {metrics['detection_rate']:.1%}")
        print(f"  same person      mean {metrics['same_person']['mean']:.3f}  "
              f"p5 {metrics['same_person']['p05']:.3f}  ({metrics['same_person']['pairs']} pairs)")
        print(f"  different person mean {metrics['different_person']['mean']:.3f}  "
              f"p95 {metrics['different_person']['p95']:.3f}  ({metrics['different_person']['pairs']} pairs)")
        print(f"  AUROC            {metrics['auroc']:.4f}")
        print(f"  threshold        {metrics['threshold']:.4f}")
        print(f"  balanced acc     {metrics['balanced_accuracy']:.4f}  "
              f"(tpr {metrics['tpr']:.4f} / fpr {metrics['fpr']:.4f})")
        print(f"  elapsed          {metrics['elapsed_s']:.1f}s")

    if metrics["auroc"] < args.fail_below:
        print(
            f"FAIL: AUROC {metrics['auroc']:.4f} below required {args.fail_below:.4f}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())