"""Result and image persistence.

Flat files on purpose. A demo's results should be inspectable with ``ls`` and diffable in
git; a database would be more code than the product needs and would make the archive
harder to read.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image

from .benchmark import BenchmarkResult, as_pil, plot_tradeoff

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data"
RESULTS = DATA / "results"
IMAGES = DATA / "images"


class ResultStore:
    def __init__(self, results: Path = RESULTS, images: Path = IMAGES) -> None:
        self.results = results
        self.images = images
        self.results.mkdir(parents=True, exist_ok=True)
        self.images.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ images

    def save_image(self, filename: str, image: object) -> str:
        target = self.images / filename
        as_pil(image).save(target)
        return f"/api/images/{filename}"

    def resolve(self, name: str) -> Path:
        """Map a stored image reference back to a path, refusing anything else."""
        candidate = (self.images / Path(name).name).resolve()
        root = self.images.resolve()
        if root not in candidate.parents and candidate != root:
            raise ValueError("path escapes image directory")
        return candidate

    # ------------------------------------------------------------ results

    def save_result(self, payload: dict) -> str:
        result_id = str(payload.get("created_at_iso", "run")).replace(":", "").replace("-", "")
        result_id = f"{result_id}_{abs(hash(payload.get('provider', ''))) % 9973:03d}"
        (self.results / f"{result_id}.json").parent.mkdir(parents=True, exist_ok=True)
        self.write(result_id, payload)
        return result_id

    def write(self, result_id: str, payload: dict) -> Path:
        target = self.results / f"{result_id}.json"
        target.write_text(json.dumps(payload, indent=2))
        return target

    def load(self, result_id: str) -> dict | None:
        target = self.results / f"{result_id}.json"
        if not target.exists():
            return None
        return json.loads(target.read_text())

    def list(self) -> list[dict]:
        summaries: list[dict] = []
        for path in sorted(self.results.glob("*.json"), reverse=True):
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            summaries.append(
                {
                    "result_id": path.stem,
                    "provider": data.get("provider"),
                    "character": data.get("character"),
                    "script_title": data.get("script_title"),
                    "created_at": data.get("created_at"),
                    "created_at_iso": data.get("created_at_iso"),
                    "recommended_strength": data.get("recommended_strength"),
                    "identity_axis_trustworthy": data.get("identity_axis_trustworthy"),
                    "auroc": (data.get("calibration") or {}).get("auroc"),
                    "elapsed_s": data.get("elapsed_s"),
                }
            )
        return summaries

    def delete(self, result_id: str) -> bool:
        target = self.results / f"{result_id}.json"
        if not target.exists():
            return False
        target.unlink()
        shutil.rmtree(self.panel_dir(result_id), ignore_errors=True)
        plot = self.results / f"{result_id}.png"
        plot.unlink(missing_ok=True)
        return True

    # ------------------------------------------------------------ panels & plots

    def panel_dir(self, result_id: str) -> Path:
        return self.results / result_id

    def panel_path(self, result_id: str, condition: str, index: int) -> Path:
        safe = condition.replace("/", "_").replace(" ", "_")
        return self.panel_dir(result_id) / f"{safe}_panel{index}.png"

    def plot_path(self, result_id: str) -> Path:
        return self.results / f"{result_id}.png"

    def plot(self, result: BenchmarkResult, result_id: str) -> str:
        """Render and save the tradeoff plot, plus a panel grid per condition."""
        out = plot_tradeoff(result, str(self.plot_path(result_id)))
        for condition in result.conditions:
            for index in range(result.panels_per_condition):
                path = self.panel_path(result_id, condition.condition, index)
                path.parent.mkdir(parents=True, exist_ok=True)
        return out
