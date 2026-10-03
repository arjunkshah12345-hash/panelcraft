"""The benchmark: sweep the identity dial, measure the tradeoff.

The product claim is a curve, not a single number, so this module's job is to produce a
curve honestly. Three measurement decisions matter:

1. **Seeds are held fixed across the sweep.** A strength that looks better must not just
   be luckier sampling.
2. **Identity and variety are measured by separate instruments.** Identity comes from a
   face embedding; variety comes from everything *outside* the face crop. Mixing them
   would let the model score well on one by cheating on the other.
3. **Calibration flags whether it was held out.** A threshold tuned on the same panels it
   is then used to judge is not a verification result, and saying otherwise would be the
   easiest lie to tell in this kind of project.

Everything here is provider-agnostic: it never assumes a face detector succeeds, and it
never assumes a provider can report a pose.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Protocol, Sequence

import numpy as np
from PIL import Image

from .compiler import Script, compile_negative_prompt, compile_prompt, panel_seed
from .identity import (
    IdentityReport,
    IdentityScorer,
    calibrate_threshold,
    get_scorer,
    score_condition,
)

FaceBox = tuple[float, float, float, float]
Progress = Callable[[str, float], None]
# Called as on_panel(condition, panel_index, image) -> a reference string to embed
# in that panel's dict. Lets callers persist images without benchmark knowing about disk.
PanelSink = Callable[[str, int, Image.Image], str | None]


def as_pil(image: object) -> Image.Image:
    """Normalise a provider image to PIL.

    Providers are not obliged to agree on a return type — the local renderer yields
    ndarrays, hosted backends yield PIL images — so every entry point here converts once
    rather than assuming.
    """
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        array = image
        if array.dtype != np.uint8:
            array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        return Image.fromarray(array[:, :, :3])
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    # Binary streams arrive from the API's base64 data-URL decoding. Decoded here rather
    # than at the call site so this stays the single place that knows how to open an image.
    if isinstance(image, io.IOBase):
        return Image.open(image).convert("RGB")
    raise TypeError(f"cannot interpret {type(image)!r} as an image")


class SupportsSweep(Protocol):
    name: str

    def generate(
        self,
        prompt: str,
        reference: Sequence[str] | Sequence[Image.Image] | None = None,
        identity_strength: float = 0.8,
        panel_index: int = 0,
        seed: int | None = None,
        negative_prompt: str | None = None,
    ): ...


# --------------------------------------------------------------------------------------
# Result shapes
# --------------------------------------------------------------------------------------


@dataclass
class TradeoffPoint:
    identity_strength: float
    identity_mean: float
    identity_min: float
    variety: float
    true_identity_distance: float | None = None
    dominated: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConditionResult:
    condition: str
    identity_strength: float
    identity_mean: float
    identity_min: float
    pairwise_mean: float
    variety: float
    true_identity_distance: float | None = None
    panels: list[dict] = field(default_factory=list)
    report: dict | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class CalibrationResult:
    threshold: float
    balanced_accuracy: float
    tpr: float
    fpr: float
    auroc: float
    n_positive: int
    n_negative: int
    held_out: bool
    caveat: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkResult:
    provider: str
    character: str
    script_title: str
    created_at: float
    strengths: list[float]
    panels_per_condition: int
    reference_images: int
    conditions: list[ConditionResult]
    tradeoff: list[TradeoffPoint]
    calibration: CalibrationResult | None
    recommended_strength: float | None
    pareto: list[float]
    elapsed_s: float
    identity_axis_trustworthy: bool = True
    identity_axis_note: str = ""
    harness_validation: dict | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        data = asdict(self)
        data["created_at_iso"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.created_at)
        )
        return data


# --------------------------------------------------------------------------------------
# Variety: everything outside the face
# --------------------------------------------------------------------------------------


def presentation_diversity(
    images: Sequence[Image.Image], face_boxes: Sequence[FaceBox | None] | None = None
) -> float:
    """Mean pairwise distance between panels *excluding* their faces.

    The face is identity, so it must be excluded or a model that repeats one face across
    six panels scores as "varied". What is left is pose, framing, lighting and set — the
    things that make a comic readable. No landmark model is needed, so this works on
    generated and stylised art where a detector frequently returns nothing.

    Range is roughly 0 (six identical panels) to 2 (maximally unrelated).
    """
    if len(images) < 2:
        return 0.0

    vectors: list[np.ndarray] = []
    for i, image in enumerate(images):
        rgb = np.asarray(as_pil(image).convert("RGB"), dtype=np.float32) / 255.0
        box = face_boxes[i] if face_boxes and i < len(face_boxes) else None
        if box is not None:
            x0, y0, x1, y1 = (int(round(v)) for v in box)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(rgb.shape[1], x1), min(rgb.shape[0], y1)
            if x1 > x0 + 2 and y1 > y0 + 2:
                rgb = rgb.copy()
                rgb[y0:y1, x0:x1] = rgb.mean(axis=(0, 1), keepdims=True)

        small = np.asarray(
            Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).resize(
                (32, 32), Image.BILINEAR
            ),
            dtype=np.float32,
        ).ravel()
        small -= small.mean()
        norm = float(np.linalg.norm(small))
        vectors.append(small / norm if norm > 1e-8 else small)

    distances = [
        float(np.linalg.norm(vectors[i] - vectors[j]))
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
    ]
    return float(np.mean(distances)) if distances else 0.0


# --------------------------------------------------------------------------------------
# Pareto analysis
# --------------------------------------------------------------------------------------


def pareto_front(points: Sequence[TradeoffPoint]) -> tuple[list[TradeoffPoint], list[TradeoffPoint]]:
    """Split points into non-dominated and dominated.

    Both axes are "higher is better", so a point is dominated when some other point is at
    least as good on identity and variety and strictly better on one of them.
    """
    front: list[TradeoffPoint] = []
    dominated_list: list[TradeoffPoint] = []
    for a in points:
        dominated = any(
            (b.identity_mean >= a.identity_mean and b.variety >= a.variety)
            and (b.identity_mean > a.identity_mean or b.variety > a.variety)
            for b in points
            if b is not a
        )
        (dominated_list if dominated else front).append(a)
    return front, dominated_list


def recommended_strength(points: Sequence[TradeoffPoint]) -> float | None:
    """Best balance on the Pareto front.

    Both axes are min-max normalised across the front first, so a strength is not chosen
    just because it wins on the axis that happens to have the wider numeric range.
    """
    if not points:
        return None
    ids = np.array([p.identity_mean for p in points], dtype=np.float64)
    var = np.array([p.variety for p in points], dtype=np.float64)

    def norm(a: np.ndarray) -> np.ndarray:
        span = a.max() - a.min()
        return np.ones_like(a) if span < 1e-12 else (a - a.min()) / span

    score = norm(ids) + norm(var)
    return float(points[int(np.argmax(score))].identity_strength)


# --------------------------------------------------------------------------------------
# References
# --------------------------------------------------------------------------------------


def build_references(
    provider: SupportsSweep,
    script: Script,
    count: int = 3,
    strength: float = 1.0,
    seed_tag: str = "ref",
) -> tuple[list[Image.Image], list[FaceBox | None], str | None]:
    """Produce the character's reference portraits and the seed that recreates them."""
    sheet_fn = getattr(provider, "reference_sheet", None)
    if callable(sheet_fn):
        try:
            sheet = sheet_fn(script.character.name, count)
        except TypeError:
            sheet = sheet_fn(script.character.name)
        images: list[Image.Image] = []
        boxes: list[FaceBox | None] = []
        for item in sheet:
            if isinstance(item, tuple):
                images.append(as_pil(item[0]))
                boxes.append(item[1] if len(item) > 1 else None)
            else:
                images.append(as_pil(item))
                boxes.append(None)
        if images:
            return images, boxes, seed_tag

    images, boxes = [], []
    for i in range(count):
        from .compiler import stable_seed

        result = provider.generate(
            prompt=compile_prompt(script, i % max(1, len(script.panels))),
            reference=None,
            identity_strength=strength,
            panel_index=i,
            seed=stable_seed(seed_tag, script.character.name, i),
            negative_prompt=compile_negative_prompt(),
        )
        images.append(as_pil(result.image))
        boxes.append(getattr(result, "face_box", None))
    return images, boxes, None


def score_panels(
    provider: SupportsSweep,
    script: Script,
    images: Sequence[object],
    face_boxes: Sequence[FaceBox | None] | None = None,
    reference_count: int = 3,
    scorer: IdentityScorer | None = None,
) -> IdentityReport:
    """Score one rendered comic against its own reference portraits.

    Shared by the API so the single-comic view and the sweep are scored by identical code —
    a number shown in the UI and a number in a stored result are then the same measurement.
    """
    scorer = scorer or get_scorer()
    refs, ref_boxes, _ = build_references(provider, script, count=reference_count)
    return score_condition(
        scorer,
        refs,
        [as_pil(i) for i in images],
        [f"panel-{i + 1}" for i in range(len(images))],
        condition="single comic",
        reference_boxes=ref_boxes,
        panel_boxes=list(face_boxes) if face_boxes else None,
    )


# --------------------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------------------


def run_sweep(
    provider: SupportsSweep,
    script: Script,
    strengths: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    panels_per_condition: int = 6,
    reference_count: int = 3,
    scorer: IdentityScorer | None = None,
    progress: Progress | None = None,
    on_panel: PanelSink | None = None,
) -> BenchmarkResult:
    """Generate every (strength, panel) pair, score identity, and build the curve."""
    started = time.time()
    scorer = scorer or get_scorer()
    strengths = [float(s) for s in strengths]
    negative = compile_negative_prompt()

    ref_images, ref_boxes, ref_seed = build_references(
        provider, script, count=reference_count
    )
    if not ref_images:
        raise ValueError("provider produced no reference portraits")

    reference_arg: Sequence[str] | Sequence[Image.Image]
    # A provider exposing ``reference_sheet`` is seed-addressable: it can rebuild the same
    # portraits from a seed, so pass the seed rather than the rendered images. The previous
    # check (``not isinstance(ref_images[0], Image.Image)``) could never be true here,
    # because ``build_references`` always returns PIL images, so every sweep silently fell
    # through to the image branch.
    if ref_seed is not None and callable(getattr(provider, "reference_sheet", None)):
        reference_arg = [ref_seed]
    else:
        reference_arg = list(ref_images)

    panel_count = max(1, len(script.panels))
    conditions: list[ConditionResult] = []
    positives: list[float] = []
    negatives: list[float] = []
    observations: list[tuple[float, float]] = []
    ground_truth: list[tuple[float, float]] = []
    measured_rows: list[tuple[float, float]] = []

    total_steps = len(strengths) * panels_per_condition
    step = 0

    for strength in strengths:
        images: list[Image.Image] = []
        boxes: list[FaceBox | None] = []
        seeds: list[int] = []

        for index in range(panels_per_condition):
            beat = index % panel_count
            seed = panel_seed(script, beat)
            seeds.append(seed)
            generated = provider.generate(
                prompt=compile_prompt(script, beat),
                reference=reference_arg,
                identity_strength=strength,
                panel_index=index,
                seed=seed,
                negative_prompt=negative,
            )
            images.append(as_pil(generated.image))
            boxes.append(getattr(generated, "face_box", None))
            step += 1
            if progress is not None:
                label = f"Generating panel {index + 1}/{panels_per_condition} at strength {strength:g}"
                progress(label, step / total_steps)

        report: IdentityReport = score_condition(
            scorer,
            ref_images,
            images,
            [f"panel-{i + 1}" for i in range(panels_per_condition)],
            condition=f"identity_strength={strength:g}",
            reference_boxes=ref_boxes,
            panel_boxes=boxes,
        )

        variety = presentation_diversity(images, boxes)
        panel_dicts = [p.to_dict() for p in report.panels]
        if on_panel is not None:
            for index, panel_dict in enumerate(panel_dicts):
                reference = on_panel(report.condition, index, images[index])
                if reference:
                    panel_dict["image_path"] = reference

        conditions.append(
            ConditionResult(
                condition=report.condition,
                identity_strength=strength,
                identity_mean=report.identity_mean,
                identity_min=report.identity_min,
                pairwise_mean=report.pairwise_mean,
                variety=variety,
                panels=panel_dicts,
                report=report.to_dict(),
            )
        )
        observations.append((report.identity_mean, variety))

        values = [p.sim_to_reference for p in report.panels]
        if strength >= max(strengths):
            positives.extend(values)
        if strength <= min(strengths):
            negatives.extend(values)

        truth_fn = getattr(provider, "ground_truth_identity_distance", None)
        if callable(truth_fn):
            truths = [
                float(truth_fn(script.character.name, strength, index))
                for index in range(panels_per_condition)
            ]
            true_distance = float(np.mean(truths))
            conditions[-1].true_identity_distance = true_distance
            ground_truth.append((strength, true_distance))
            for value, truth in zip(values, truths):
                measured_rows.append((truth, value))

    tradeoff = [
        TradeoffPoint(
            identity_strength=c.identity_strength,
            identity_mean=c.identity_mean,
            identity_min=c.identity_min,
            variety=c.variety,
            true_identity_distance=c.true_identity_distance,
        )
        for c in conditions
    ]

    front, dominated = pareto_front(tradeoff)
    for point in tradeoff:
        point.dominated = any(point is d for d in dominated)

    calibration: CalibrationResult | None = None
    if positives and negatives:
        tuned = calibrate_threshold(positives, negatives)
        calibration = CalibrationResult(
            threshold=tuned["threshold"],
            balanced_accuracy=tuned["balanced_accuracy"],
            tpr=tuned["tpr"],
            fpr=tuned["fpr"],
            auroc=tuned["auroc"],
            n_positive=len(positives),
            n_negative=len(negatives),
            held_out=False,
            caveat=(
                "Threshold tuned on the strongest and weakest panels of this same sweep. "
                "It is a fit to these panels, not an independent verification result. "
                "See scripts/validate_scorer.py for the held-out scorer check."
            ),
        )

    # A face embedding model is trained on photographs. On flat vector art it mostly reads
    # pose and rendering style as if they were identity, so the identity axis is flagged
    # untrustworthy rather than quietly published as a result.
    caps = getattr(provider, "capabilities", None)
    caps = caps() if callable(caps) else caps
    domain = getattr(caps, "measurement_domain", None) or "photographic"
    has_truth = bool(getattr(caps, "provides_ground_truth", False))
    identity_ok = domain == "photographic"
    identity_note = (
        "Face embedding models are trained on photographs. On this provider's synthetic "
        "vector faces the identity axis tracks pose and rendering style rather than "
        "identity, so treat identity values here as pipeline smoke-test output, not as a "
        "measurement. The variety axis is sound, and identity is measured properly on "
        "real output (see scripts/validate_scorer.py and the hosted providers)."
        if not identity_ok
        else "Face embeddings measured on photographic output."
    )

    # Does the measurement track a known truth? Where the provider exposes ground truth
    # this is the only honest way to judge the instrument on this material.
    validation: dict | None = None
    if len(measured_rows) >= 6:
        truths = np.array([t for t, _ in measured_rows], dtype=np.float64)
        measured = np.array([m for _, m in measured_rows], dtype=np.float64)
        try:
            from scipy.stats import spearmanr

            rho, pvalue = spearmanr(truths, measured)
            validation = {
                "n": int(len(truths)),
                "spearman_true_drift_vs_measured_similarity": round(float(rho), 4),
                "p_value": float(f"{pvalue:.3g}"),
                "reading": (
                    "Measurement tracks known identity drift."
                    if rho < -0.5
                    else "Measurement does not reliably track known identity drift."
                ),
            }
        except Exception:
            validation = None

    recommendation = recommended_strength(front)
    if not identity_ok:
        # Never quote a recommendation derived from an axis we just declared invalid.
        recommendation = None
    notes = [
        "Seeds are identical across the sweep, so identity strength is the only variable.",
        "Variety excludes each panel's own face crop, so repeated faces cannot inflate it.",
        f"References: {len(ref_images)} portrait(s), seed {ref_seed or 'n/a'}.",
        f"Measurement domain: {domain}.",
    ]
    if not identity_ok:
        notes.append(
            "Identity axis flagged untrustworthy for this provider (see identity_axis_note). "
            "No recommended strength is emitted, because it would be derived from that axis."
        )
    if validation is not None:
        notes.append(
            f"Harness check against ground truth: Spearman "
            f"{validation['spearman_true_drift_vs_measured_similarity']} "
            f"(p={validation['p_value']}, n={validation['n']})."
        )
    if ground_truth:
        notes.append(
            "Provider exposes ground-truth drift, so the tradeoff curve can also be read "
            "without trusting the measurement: true identity distance "
            + ", ".join(f"s={s:g}->{d:.3f}" for s, d in ground_truth)
            + "."
        )
    if scorer.detector_kind != "mtcnn":
        notes.append(
            f"Face detection resolved via {scorer.detector_kind!r}; provider-supplied face "
            "boxes were used where available. Real diffusion output should be "
            "detector-resolved."
        )


    return BenchmarkResult(
        provider=provider.name,
        character=script.character.name,
        script_title=script.title,
        created_at=time.time(),
        strengths=strengths,
        panels_per_condition=panels_per_condition,
        reference_images=len(ref_images),
        conditions=conditions,
        tradeoff=tradeoff,
        calibration=calibration,
        recommended_strength=recommendation,
        pareto=[p.identity_strength for p in front],
        elapsed_s=time.time() - started,
        identity_axis_trustworthy=identity_ok,
        identity_axis_note=identity_note,
        harness_validation=validation,
        notes=notes,
    )


# --------------------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------------------


def plot_tradeoff(result: BenchmarkResult, path: str) -> str:
    """Identity against variety, with the dominated region greyed out."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = result.tradeoff
    identity = [p.identity_mean for p in points]
    variety = [p.variety for p in points]
    strengths = [p.identity_strength for p in points]
    front_mask = [not p.dominated for p in points]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), facecolor="#fbfbf8")
    ink, brand, muted = "#14110f", "#0566ff", "#b9b2a6"

    for axis in axes:
        axis.set_facecolor("#fbfbf8")
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            axis.spines[spine].set_color(muted)
        axis.tick_params(colors=ink, labelsize=9)

    ax = axes[0]
    ax.plot(
        variety,
        identity,
        color=muted,
        lw=1.6,
        ls="--",
        zorder=1,
        label="measured tradeoff",
    )
    ax.scatter(
        [v for v, f in zip(variety, front_mask) if f],
        [i for i, f in zip(identity, front_mask) if f],
        s=130,
        color=brand,
        zorder=3,
        edgecolor=ink,
        linewidth=1.1,
        label="non-dominated",
    )
    ax.scatter(
        [v for v, f in zip(variety, front_mask) if not f],
        [i for i, f in zip(identity, front_mask) if not f],
        s=90,
        color="#efece5",
        zorder=2,
        edgecolor=muted,
        linewidth=1.1,
        label="dominated",
    )
    for s, v, i in zip(strengths, variety, identity):
        ax.annotate(
            f"{s:g}",
            (v, i),
            textcoords="offset points",
            xytext=(9, 7),
            fontsize=9,
            color=ink,
            fontweight="bold",
        )

    if result.recommended_strength is not None:
        ax.set_title(
            f"Recommended identity strength: {result.recommended_strength:g}",
            color=ink,
            fontsize=12,
            pad=14,
            fontweight="bold",
        )
    ax.set_xlabel("panel variety (face excluded)  →  more", color=ink, fontsize=10)
    ax.set_ylabel(
        "identity consistency  →  more" if result.identity_axis_trustworthy
        else "identity (UNTRUSTWORTHY on synthetic art)  →  more",
        color=ink if result.identity_axis_trustworthy else "#b4472e",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=ink)

    ax = axes[1]
    width = 0.36
    positions = np.arange(len(points))
    ax.bar(
        positions - width / 2,
        identity,
        width,
        color=brand,
        label="identity mean",
    )
    ax.bar(
        positions + width / 2,
        variety,
        width,
        color="#cfd8e8",
        label="variety",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{s:g}" for s in strengths])
    ax.set_xlabel("identity strength", color=ink, fontsize=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=ink)

    cal = result.calibration
    if cal is not None:
        subtitle = (
            f"AUROC {cal.auroc:.3f} (in-sweep, not held out)  ·  "
            f"threshold {cal.threshold:.3f}  ·  {result.provider}  ·  "
            f"{result.panels_per_condition} panels/condition"
        )
    else:
        subtitle = f"{result.provider}  ·  {result.panels_per_condition} panels/condition"
    fig.suptitle(
        "PanelCraft — identity consistency against panel variety",
        color=ink,
        fontsize=14.5,
        x=0.055,
        ha="left",
        fontweight="bold",
        y=0.985,
    )
    fig.text(0.055, 0.925, subtitle, color="#6b635a", fontsize=9.5, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path