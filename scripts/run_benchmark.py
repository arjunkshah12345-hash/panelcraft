#!/usr/bin/env python
"""Run a PanelCraft sweep from the terminal and write the results to disk.

    python scripts/run_benchmark.py --provider synthetic
    python scripts/run_benchmark.py --provider pulid --panels 6   # needs REPLICATE_API_TOKEN

Saves ``results/<id>.json``, a tradeoff plot, and every panel as a PNG so a run can be
inspected, committed, and cited.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.services.benchmark import as_pil, plot_tradeoff, run_sweep
from api.services.compiler import default_script
from api.services.providers.registry import available_providers, get_provider


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="synthetic")
    parser.add_argument("--strengths", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--panels", type=int, default=6)
    parser.add_argument("--references", type=int, default=3)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    strengths = [float(s) for s in args.strengths.split(",") if s.strip()]

    try:
        provider = get_provider(args.provider)
    except Exception as exc:
        print(f"provider {args.provider!r} unavailable: {exc}", file=sys.stderr)
        print("available:", [c["name"] for c in available_providers()], file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def progress(label: str, fraction: float) -> None:
        bar = "#" * int(fraction * 28)
        sys.stderr.write(f"\r  [{bar:<28}] {fraction * 100:5.1f}%  {label[:44]:<44}")
        sys.stderr.flush()

    # The result id is derived up front so panels stream straight to their final location
    # as they are generated: a long sweep never holds every image in memory, and a crash
    # keeps whatever already finished.
    result_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    def save_panel(condition: str, index: int, image) -> str | None:
        safe = condition.replace(" ", "_").replace("/", "_")
        target = out_dir / result_id / safe / f"panel{index}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        as_pil(image).save(target)
        return str(target)

    print(f"provider={provider.name} strengths={strengths} panels={args.panels}")
    result = run_sweep(
        provider,
        default_script(),
        strengths=strengths,
        panels_per_condition=args.panels,
        reference_count=args.references,
        progress=progress,
        on_panel=save_panel,
    )
    sys.stderr.write("\n")

    payload = result.as_dict()
    json_path = out_dir / f"{result_id}.json"
    json_path.write_text(json.dumps(payload, indent=2))
    plot_path = plot_tradeoff(result, str(out_dir / f"{result_id}.png"))

    print(f"\nidentity trustworthy: {result.identity_axis_trustworthy}")
    print(f"recommended strength: {result.recommended_strength}")
    if result.harness_validation:
        v = result.harness_validation
        print(
            f"harness vs ground truth: spearman "
            f"{v['spearman_true_drift_vs_measured_similarity']} (p={v['p_value']})"
        )
    print(f"{'strength':>8} {'identity':>9} {'variety':>8} {'true drift':>11}  note")
    for point in result.tradeoff:
        drift = (
            f"{point.true_identity_distance:>11.3f}"
            if point.true_identity_distance is not None
            else f"{'-':>11}"
        )
        print(
            f"{point.identity_strength:>8.2f} {point.identity_mean:>9.3f} "
            f"{point.variety:>8.3f} {drift}  {'dominated' if point.dominated else ''}"
        )
    if result.calibration:
        c = result.calibration
        print(f"\ncalibration (in-sweep, held_out={c.held_out}): "
              f"threshold={c.threshold:.3f} balanced_acc={c.balanced_accuracy:.3f} "
              f"auroc={c.auroc:.3f}")
    for note in result.notes:
        print(f"  - {note}")

    print(f"\nwrote {json_path}")
    print(f"wrote {plot_path}")
    print(f"wrote panels under {out_dir / result_id}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())