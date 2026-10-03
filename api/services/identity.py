"""Identity scoring: turn generated panels into a measurable consistency number.

The whole project rests on this module. If identity consistency is a vibe, we have
a demo. If it is a 512-dimensional embedding you can put a cosine similarity on, we
have a benchmark.

Design notes that matter:

* FaceNet (vggface2 weights) is the primary scorer. It is MIT-licensed, which keeps
  the hackathon repo legally clean for anyone who clones it. InsightFace's ArcFace is
  the accuracy king but is research-licensed, so it is opt-in via backends.arcface.
* Face *detection* is deliberately allowed to fail. Generated faces are frequently
  off-distribution for a detector trained on photographs, and a pipeline that dies
  with "no face found" is useless. We fall back to a centre crop, which works well
  enough on stylised or centred portraits.
* Everything is synchronous torch. Callers run it in a worker thread.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# Cosine similarity below this is "a different person" for a face verification model
# trained with hard negatives. Tunable per-threshold per project.
DEFAULT_THRESHOLD = 0.55

BackendName = Literal["facenet", "arcface"]


def _load_rgb(source: str | Path | Image.Image | np.ndarray) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, np.ndarray):
        return Image.fromarray(source.astype("uint8")).convert("RGB")
    return Image.open(source).convert("RGB")


@dataclass
class PanelScore:
    """How well one panel matches the reference character."""

    panel_id: str
    sim_to_reference: float
    nearest_reference: str | None = None
    detected_face: bool = True
    note: str | None = None
    threshold: float = DEFAULT_THRESHOLD

    @property
    def verdict(self) -> str:
        return "consistent" if self.sim_to_reference >= self.threshold else "drift"

    def to_dict(self) -> dict:
        return {
            "panel_id": self.panel_id,
            "sim_to_reference": round(self.sim_to_reference, 4),
            "nearest_reference": self.nearest_reference,
            "detected_face": self.detected_face,
            "verdict": self.verdict,
            "note": self.note,
            "threshold": round(self.threshold, 4),
        }


@dataclass
class IdentityReport:
    """Aggregate consistency measurement for one condition (one identity strength)."""

    condition: str
    identity_mean: float
    identity_min: float
    identity_std: float
    pairwise_mean: float
    pairwise_min: float
    matrix: list[list[float]]
    panel_ids: list[str]
    panels: list[PanelScore] = field(default_factory=list)
    reference_count: int = 0
    detection_rate: float = 1.0
    threshold: float = DEFAULT_THRESHOLD
    notes: list[str] = field(default_factory=list)

    @property
    def consistent(self) -> bool:
        """Every panel clears the threshold, not just the average.

        A comic that is on-model in five panels and off-model in the sixth still reads as
        a different character, so the worst panel is the one that decides.
        """
        return self.identity_min >= self.threshold

    def to_dict(self) -> dict:
        return {
            "condition": self.condition,
            "identity_mean": round(self.identity_mean, 4),
            "identity_min": round(self.identity_min, 4),
            "identity_std": round(self.identity_std, 4),
            "pairwise_mean": round(self.pairwise_mean, 4),
            "pairwise_min": round(self.pairwise_min, 4),
            "matrix": [[round(v, 4) for v in row] for row in self.matrix],
            "panel_ids": self.panel_ids,
            "panels": [
                {
                    "panel_id": p.panel_id,
                    "sim_to_reference": round(p.sim_to_reference, 4),
                    "nearest_reference": p.nearest_reference,
                    "detected_face": p.detected_face,
                    "verdict": p.verdict,
                    "note": p.note,
                    "threshold": round(p.threshold, 4),
                }
                for p in self.panels
            ],
            "reference_count": self.reference_count,
            "detection_rate": round(self.detection_rate, 4),
            "threshold": round(self.threshold, 4),
            "consistent": self.consistent,
            "notes": self.notes,
        }


class IdentityScorer:
    """Loads a face embedding model once and scores images against a reference set."""

    def __init__(
        self,
        backend: BackendName = "facenet",
        device: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        import torch

        self.backend: BackendName = backend
        self.threshold = threshold
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self._model = None
        self._detector = None
        self._torch = torch
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- model

    def _ensure_model(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            torch = self._torch
            if self.backend == "facenet":
                from facenet_pytorch import InceptionResnetV1

                model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
            elif self.backend == "arcface":
                # Research-licensed. Opt in only if you accept the terms.
                from insightface.model_zoo import get_model

                model = get_model("buffalo_l")
                model = model.to(self.device)
            else:  # pragma: no cover - guarded by Literal
                raise ValueError(f"unknown backend {self.backend!r}")
            self._model = model
            log.info("identity model loaded: backend=%s device=%s", self.backend, self.device)

    @property
    def detector_kind(self) -> str:
        """Which detector this scorer ended up with, resolved lazily."""
        self._ensure_detector()
        return self._detector_kind

    def _ensure_detector(self):
        if self._detector is not None:
            return
        with self._lock:
            if self._detector is not None:
                return
            try:
                from facenet_pytorch import MTCNN

                self._detector = MTCNN(keep_all=True, device=self.device)
                self._detector_kind = "mtcnn"
            except Exception:  # pragma: no cover - detection is best effort
                log.warning("MTCNN unavailable, falling back to centre crop only")
                self._detector = None
                self._detector_kind = "none"

    # ------------------------------------------------------------- embedding

    def _crop_face(
        self, img: Image.Image, face_box: tuple[int, int, int, int] | None = None
    ) -> tuple[Image.Image, bool]:
        """Return a face crop and whether a detector was used.

        An explicit ``face_box`` from the generator wins: it is ground truth, and
        detection on stylised or non-photographic output is the flakiest link in this
        chain. Detection remains the fallback for real diffusion output, where the
        provider has no idea where the face landed.
        """
        if face_box is not None:
            x1, y1, x2, y2 = (int(v) for v in face_box)
            x1, y1 = max(0, min(x1, img.width - 2)), max(0, min(y1, img.height - 2))
            x2, y2 = max(x1 + 2, min(x2, img.width)), max(y1 + 2, min(y2, img.height))
            return img.crop((x1, y1, x2, y2)), True

        self._ensure_detector()
        if self._detector is None:
            return self._center_crop(img), False
        try:
            boxes = self._detector.detect(
                _np_rgb(img), landmarks=False
            )
            if boxes is not None and len(boxes) > 0:
                box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                x1, y1, x2, y2 = [int(v) for v in box[:4]]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 - x1 > 16 and y2 - y1 > 16:
                    return img.crop((x1, y1, x2, y2)), True
        except Exception as exc:  # pragma: no cover
            log.debug("detection failed: %s", exc)
        return self._center_crop(img), False

    @staticmethod
    def _center_crop(img: Image.Image) -> Image.Image:
        side = min(img.width, img.height)
        left = (img.width - side) // 2
        top = (img.height - side) // 2
        return img.crop((left, top, left + side, top + side))

    def embed(
        self,
        source: str | Path | Image.Image | np.ndarray,
        *,
        detect: bool = True,
        face_box: tuple[int, int, int, int] | None = None,
    ) -> tuple[np.ndarray, bool]:
        """Embed one image. Returns (512-d L2-normalised vector, face_was_found)."""
        torch = self._torch
        self._ensure_model()
        img = _load_rgb(source)
        if face_box is not None:
            crop, detected = self._crop_face(img, face_box)
        elif detect:
            crop, detected = self._crop_face(img)
        else:
            crop, detected = self._center_crop(img), False

        if self.backend == "facenet":
            from torchvision.transforms import functional as TF

            tensor = TF.to_tensor(crop)
            tensor = TF.resize(tensor, [160, 160])
            tensor = (tensor - 0.5) / 0.5
            batch = tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self._model(batch)
            emb = torch.nn.functional.normalize(emb, dim=1)
            return emb[0].cpu().numpy().astype(np.float32), detected

        # InsightFace path expects an aligned-ish BGR crop.
        arr = _np_rgb(crop)[:, :, ::-1]
        face = self._model.get(arr)
        if face is None:  # pragma: no cover
            return np.zeros(512, dtype=np.float32), False
        emb = face.normed_embedding
        return np.asarray(emb, dtype=np.float32), detected

    def embed_many(
        self,
        sources: Sequence[str | Path | Image.Image | np.ndarray],
        *,
        detect: bool = True,
        face_boxes: Sequence[tuple[int, int, int, int] | None] | None = None,
    ) -> tuple[np.ndarray, list[bool]]:
        vectors: list[np.ndarray] = []
        found: list[bool] = []
        for i, src in enumerate(sources):
            box = face_boxes[i] if face_boxes is not None and i < len(face_boxes) else None
            vec, ok = self.embed(src, detect=detect, face_box=box)
            vectors.append(vec)
            found.append(ok)
        return np.stack(vectors), found


# --------------------------------------------------------------- measurements


def _unit(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-9, None)


def center_embeddings(anchor: np.ndarray, *groups: np.ndarray) -> list[np.ndarray]:
    """Remove the common-mode style component, using a *fixed* anchor.

    FaceNet is trained on photographs. Every panel in a run shares a rendering style,
    background treatment, crop and lighting rig, and that shared component dominates the
    raw embedding: strangers still score 0.8 against a reference and the measurement has
    almost no dynamic range. Subtracting a common mean leaves the variance that actually
    encodes identity.

    The anchor **must** be identical for every condition, so the transform is
    comparable across conditions. Centering each condition on its own mean silently
    destroys the comparison, because every condition is then centred on its own answer.
    We anchor on the reference portraits, which are constant for a given character.
    """
    mu = anchor.mean(axis=0, keepdims=True)
    return [_unit(g - mu) for g in groups]


def cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity. Inputs are assumed L2-normalised."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.clip(norms, 1e-9, None)
    return normed @ normed.T


def pairwise_stats(matrix: np.ndarray) -> tuple[float, float]:
    """Mean and minimum off-diagonal similarity from a self-similarity matrix."""
    n = matrix.shape[0]
    if n < 2:
        return float(matrix[0][0]), float(matrix[0][0])
    iu = np.triu_indices(n, k=1)
    vals = matrix[iu]
    return float(vals.mean()), float(vals.min())


def score_condition(
    scorer: IdentityScorer,
    reference_images: Sequence[str | Path | Image.Image | np.ndarray],
    panel_images: Sequence[str | Path | Image.Image | np.ndarray],
    panel_ids: Sequence[str] | None = None,
    condition: str = "condition",
    *,
    reference_names: Sequence[str] | None = None,
    reference_boxes: Sequence[tuple[int, int, int, int] | None] | None = None,
    panel_boxes: Sequence[tuple[int, int, int, int] | None] | None = None,
    center: bool = True,
    threshold: float = DEFAULT_THRESHOLD,
) -> IdentityReport:
    """Measure one experimental condition against a fixed reference set.

    ``identity_mean`` is the mean over panels of each panel's *best* similarity to any
    reference photo. Best-of-K rather than mean-of-K is deliberate: a character
    legitimately looks different across reference angles, so we ask "does this panel
    look like this character at all", not "does it match this exact photo".
    """
    ids = list(panel_ids) if panel_ids is not None else [f"p{i}" for i in range(len(panel_images))]
    notes: list[str] = []

    ref_emb, _ = scorer.embed_many(reference_images, face_boxes=reference_boxes)
    panel_emb, detected = scorer.embed_many(panel_images, face_boxes=panel_boxes)

    if center:
        # First positional arg is the anchor; it is NOT returned as a group.
        ref_emb, panel_emb = center_embeddings(ref_emb, ref_emb, panel_emb)

    cross = cosine_matrix(np.vstack([ref_emb, panel_emb]))[len(ref_emb):, : len(ref_emb)]

    panel_scores: list[PanelScore] = []
    for idx, pid in enumerate(ids):
        sims = cross[idx]
        best = int(np.argmax(sims))
        panel_scores.append(
            PanelScore(
                panel_id=pid,
                sim_to_reference=float(sims[best]),
                nearest_reference=(
                    reference_names[best]
                    if reference_names and best < len(reference_names)
                    else f"ref{best}"
                ),
                detected_face=bool(detected[idx]),
                note=None if detected[idx] else "detector miss, scored on centre crop",
                threshold=threshold,
            )
        )

    sims_to_ref = np.array([p.sim_to_reference for p in panel_scores], dtype=np.float32)
    matrix = cosine_matrix(panel_emb)
    pair_mean, pair_min = pairwise_stats(matrix)
    detection_rate = float(np.mean(detected)) if detected else 0.0
    if detection_rate < 1.0:
        notes.append(
            f"{1 - detection_rate:.0%} of panels had no detectable face; "
            "scored on centre crop which is slightly pessimistic"
        )

    return IdentityReport(
        condition=condition,
        identity_mean=float(sims_to_ref.mean()) if len(sims_to_ref) else 0.0,
        identity_min=float(sims_to_ref.min()) if len(sims_to_ref) else 0.0,
        identity_std=float(sims_to_ref.std()) if len(sims_to_ref) else 0.0,
        pairwise_mean=pair_mean,
        pairwise_min=pair_min,
        matrix=matrix.tolist(),
        panel_ids=ids,
        panels=panel_scores,
        reference_count=len(reference_images),
        detection_rate=detection_rate,
        threshold=threshold,
        notes=notes,
    )


# ------------------------------------------------------------------ singletons


@lru_cache(maxsize=4)
def get_scorer(backend: BackendName = "facenet", threshold: float = DEFAULT_THRESHOLD) -> IdentityScorer:
    """Process-wide scorer. Model weights load once, not per request."""
    return IdentityScorer(backend=backend, threshold=threshold)


def _np_rgb(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"))

# ------------------------------------------------------------- calibration


def auroc(positives: Iterable[float], negatives: Iterable[float]) -> float:
    """Probability a random positive outranks a random negative.

    Ties count as half. This is the headline number for the harness: it says whether
    the measurement separates "same character" from "different character" at all.
    """
    pos = np.asarray(list(positives), dtype=np.float64)
    neg = np.asarray(list(negatives), dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1)
    # Average ranks over tied values so ties contribute exactly 0.5.
    values = np.concatenate([pos, neg])
    sorted_vals = values[order]
    i = 0
    while i < sorted_vals.size:
        j = i
        while j + 1 < sorted_vals.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    rank_sum = ranks[: pos.size].sum()
    return float((rank_sum - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def calibrate_threshold(positives: Sequence[float], negatives: Sequence[float]) -> dict:
    """Pick a decision threshold from measured data instead of guessing one.

    Sweeps every candidate cut point and keeps the one maximising balanced accuracy,
    which is the right objective when positives and negatives are unbalanced (they are:
    a good condition is mostly correct panels with a few failures).
    """
    pos = np.asarray(list(positives), dtype=np.float64)
    neg = np.asarray(list(negatives), dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return {"threshold": DEFAULT_THRESHOLD, "balanced_accuracy": float("nan"), "auroc": auroc(pos, neg)}

    candidates = np.unique(np.concatenate([pos, neg]))
    candidates = np.concatenate([[candidates.min() - 1e-6], candidates, [candidates.max() + 1e-6]])
    best_threshold, best_score = DEFAULT_THRESHOLD, -1.0
    for t in candidates:
        tpr = float((pos >= t).mean())
        tnr = float((neg < t).mean())
        score = 0.5 * (tpr + tnr)
        if score > best_score:
            best_score, best_threshold = score, float(t)
    return {
        "threshold": round(best_threshold, 4),
        "balanced_accuracy": round(best_score, 4),
        "tpr": round(float((pos >= best_threshold).mean()), 4),
        "fpr": round(float((neg >= best_threshold).mean()), 4),
        "auroc": round(auroc(pos, neg), 4),
    }
