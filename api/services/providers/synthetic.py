"""Deterministic synthetic character renderer.

This is not a toy. It exists for three reasons:

1. **The harness must be runnable with zero API keys**, on day one, before any
   generation budget is spent. If the measurement pipeline can't be trusted, nothing
   else matters.
2. **It makes the tradeoff reproducible.** Two people running the sweep get identical
   numbers, which is exactly what a benchmark needs.
3. **It is a ground-truth generator.** Identity drift here is a number we control, so we
   can verify the scorer actually detects drift when drift is present. A scorer that
   cannot see a known 40% drift is broken, and we find out without spending a cent.

The renderer maps an identity parameter vector to facial geometry, then mixes that
vector with a per-panel random draw according to ``identity_strength``. At strength 1
every panel shares one face; at 0 every panel is a stranger. Pose freedom collapses as
identity is pinned, reproducing the stiff-panels failure mode of real ID-conditioned
diffusion models.

The same Provider interface is implemented by the real PuLID/InstantID backends, so
every consumer is provider-agnostic and nothing downstream knows the difference.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from .base import GeneratedPanel, ProviderCapabilities

PANEL_W, PANEL_H = 768, 1024
SUPERSAMPLE = 3

# Identity parameters. Order is fixed because we hash these vectors.
IDENTITY_FIELDS = (
    "head_width",
    "head_height",
    "jaw_taper",
    "eye_spacing",
    "eye_width",
    "eye_height",
    "eye_depth",
    "brow_angle",
    "brow_thickness",
    "brow_height",
    "nose_width",
    "nose_length",
    "mouth_width",
    "mouth_height",
    "lip_fullness",
    "ear_size",
    "cheek_fullness",
    "chin_length",
    "skin_tone",
    "hair_style",
    "hair_volume",
    "hair_hue",
)

# Ranges are deliberately wider than realistic human variation. Two strangers drawn
# from these ranges must land far apart in embedding space, otherwise the whole
# measurement has no dynamic range and every panel clears any sensible threshold.
IDENTITY_RANGES: dict[str, tuple[float, float]] = {
    "head_width": (0.74, 1.28),
    "head_height": (0.80, 1.22),
    "jaw_taper": (0.0, 1.0),
    "eye_spacing": (0.78, 1.24),
    "eye_width": (0.74, 1.30),
    "eye_height": (0.68, 1.36),
    "eye_depth": (0.0, 1.0),
    "brow_angle": (-0.55, 0.55),
    "brow_thickness": (0.5, 1.9),
    "brow_height": (-0.45, 0.45),
    "nose_width": (0.55, 1.50),
    "nose_length": (0.68, 1.40),
    "mouth_width": (0.62, 1.42),
    "mouth_height": (0.55, 1.50),
    "lip_fullness": (0.2, 1.4),
    "ear_size": (0.55, 1.50),
    "cheek_fullness": (0.58, 1.42),
    "chin_length": (0.68, 1.38),
    "skin_tone": (0.0, 1.0),
    "hair_style": (0.0, 5.0),
    "hair_volume": (0.45, 1.7),
    "hair_hue": (0.0, 1.0),
}


@dataclass(frozen=True)
class Identity:
    """A character's face as a point in identity space."""

    head_width: float
    head_height: float
    jaw_taper: float
    eye_spacing: float
    eye_width: float
    eye_height: float
    eye_depth: float
    brow_angle: float
    brow_thickness: float
    brow_height: float
    nose_width: float
    nose_length: float
    mouth_width: float
    mouth_height: float
    lip_fullness: float
    ear_size: float
    cheek_fullness: float
    chin_length: float
    skin_tone: float
    hair_style: float
    hair_volume: float
    hair_hue: float

    def as_vector(self) -> np.ndarray:
        return np.array([getattr(self, f) for f in IDENTITY_FIELDS], dtype=np.float32)

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "Identity":
        values = np.clip(
            vector,
            [IDENTITY_RANGES[f][0] for f in IDENTITY_FIELDS],
            [IDENTITY_RANGES[f][1] for f in IDENTITY_FIELDS],
        )
        return cls(**{f: float(v) for f, v in zip(IDENTITY_FIELDS, values)})

    @classmethod
    def random(cls, rng: np.random.Generator) -> "Identity":
        vector = np.array(
            [rng.uniform(IDENTITY_RANGES[f][0], IDENTITY_RANGES[f][1]) for f in IDENTITY_FIELDS],
            dtype=np.float32,
        )
        return cls.from_vector(vector)


@dataclass(frozen=True)
class Appearance:
    """Per-panel presentation. This is what *should* vary between panels."""

    yaw: float  # -1 profile left .. +1 profile right
    pitch: float  # -1 looking up .. +1 looking down
    smile: float  # -1 .. +1
    brow_raise: float  # -1 .. +1
    eye_open: float  # 0 .. 1
    zoom: float  # 0.85 wide .. 1.2 close-up
    light_dir: float  # -1 .. +1
    background_hue: float  # 0 .. 1
    brightness: float  # 0.8 .. 1.15


def _stable_seed(*parts: object) -> int:
    """Deterministic across processes. Python's hash() is salted, so we use sha256."""
    joined = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(joined).digest()[:8], "big")


def _character_seed(reference: Sequence[object] | None, prompt: str) -> str:
    """Stable identity for a reference argument, used to seed every random draw.

    A reference can arrive as a seed string or as a rendered image. Strings pass through,
    but images must be hashed by pixel content: ``str(PIL.Image)`` embeds the object's
    memory address, so stringifying a reference image produced a different character on
    every call. That made each sweep render an entirely different cast, which is exactly
    the kind of uncontrolled variable this harness exists to eliminate.
    """
    if not reference:
        return prompt
    first = reference[0]
    if isinstance(first, (tuple, list)):  # provider protocol allows (image, face_box) pairs
        return _character_seed(list(first[:1]), prompt)
    if isinstance(first, str):
        return first
    if isinstance(first, np.ndarray):
        return "img:" + hashlib.sha256(np.ascontiguousarray(first).tobytes()).hexdigest()[:32]
    tobytes = getattr(first, "tobytes", None)
    if callable(tobytes):
        return "img:" + hashlib.sha256(tobytes()).hexdigest()[:32]
    return str(first)


def _hsv(hue01: float, sat: float, val: float) -> tuple[int, int, int]:
    hue = int(np.clip(hue01 % 1.0, 0.0, 0.999) * 179)
    return hue, int(np.clip(sat, 0, 1) * 255), int(np.clip(val, 0, 1) * 255)


class SyntheticRenderer:
    """Renders a face from an Identity vector plus an Appearance vector."""

    def render(
        self,
        identity: Identity,
        appearance: Appearance,
        *,
        caption: str = "",
        with_caption: bool = False,
    ) -> np.ndarray:
        S = SUPERSAMPLE
        w, h = PANEL_W * S, PANEL_H * S
        geom = self._geometry(identity, appearance, w, h)
        skin = self._skin_colour(identity.skin_tone, appearance.light_dir, appearance.brightness)

        # Two-pass rendering: soft volumetric paint underneath, crisp ink on top.
        # Detectors trained on photographs reject hard-edged vector art, and the whole
        # measurement stack depends on them working. This is load-bearing, not styling.
        paint = np.zeros_like(self._background(appearance, w, h))
        self._draw_shoulders(paint, identity, skin, appearance, geom)
        self._draw_ears(paint, identity, skin, geom)
        self._draw_head(paint, identity, skin, appearance, geom)
        self._draw_brows(paint, identity, appearance, skin, geom)
        self._draw_eyes(paint, identity, appearance, skin, geom)
        self._draw_nose(paint, identity, appearance, skin, geom)
        self._draw_mouth(paint, identity, appearance, geom)
        self._draw_hair(paint, identity, appearance, geom)
        paint = _soften(paint, 1.6 * S)

        canvas = self._background(appearance, w, h)
        # Soft alpha from the paint layer so shading bleeds rather than tiles.
        alpha = _layer_alpha(paint, w, h)
        canvas = (canvas.astype(np.float32) * (1 - alpha) + paint.astype(np.float32) * alpha)

        ink = np.zeros_like(canvas)
        self._draw_ink(ink, identity, appearance, skin, geom)
        canvas = canvas * (1 - 0.75 * _layer_alpha(ink, w, h)) + ink * (0.75 * _layer_alpha(ink, w, h))
        canvas = _vignette(np.clip(canvas, 0, 255).astype(np.uint8), 0.28)

        image = cv2.resize(
            canvas.astype(np.uint8), (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA
        )
        if with_caption and caption:
            image = self._draw_caption(image, caption)
        return image

    def face_box(self, ident: Identity, ap: Appearance) -> tuple[int, int, int, int]:
        """Ground-truth head region in final panel pixels.

        The renderer knows exactly where it put the face, so we hand that to the
        scorer rather than asking a detector trained on photographs to guess. Saves a
        step and removes a failure mode on stylised output.
        """
        g = self._geometry(ident, ap, PANEL_W * SUPERSAMPLE, PANEL_H * SUPERSAMPLE)
        cx, cy, hw, hh = g["cx"], g["cy"], g["head_w"], g["head_h"]
        # Inner-face region only: brow line to chin, cheek to cheek. Hair and clothing
        # are deliberately excluded because they dominate the embedding otherwise, and
        # hair_style/hair_hue would then masquerade as identity.
        box = (cx - hw * 0.95, cy - hh * 0.62, cx + hw * 0.95, cy + hh * 0.95)
        scale = SUPERSAMPLE
        x1, y1, x2, y2 = [int(v / scale) for v in box]
        x1 = max(0, min(PANEL_W - 2, x1))
        y1 = max(0, min(PANEL_H - 2, y1))
        x2 = max(x1 + 2, min(PANEL_W, x2))
        y2 = max(y1 + 2, min(PANEL_H, y2))
        return (x1, y1, x2, y2)

    def _geometry(self, ident: Identity, ap: Appearance, w: int, h: int) -> dict:
        """All derived coordinates in one place, so paint and ink passes agree."""
        head_w = int(w * 0.255 * ident.head_width * ap.zoom)
        head_h = int(head_w * 1.32 * ident.head_height)
        cx = w // 2 + int(ap.yaw * head_w * 0.16)
        cy = int(h * 0.42 + ap.pitch * head_h * 0.08)
        eye_y = cy - int(head_h * 0.10)
        return {
            "head_w": head_w,
            "head_h": head_h,
            "cx": cx,
            "cy": cy,
            "eye_y": eye_y,
            "eye_spacing": head_w * ident.eye_spacing * 0.50,
            "eye_w": int(head_w * 0.30 * ident.eye_width),
            "eye_h": max(3, int(head_h * 0.115 * ident.eye_height * ap.eye_open)),
            "brow_y": cy - int(head_h * (0.21 + ident.brow_height * 0.06)) - int(ap.brow_raise * head_h * 0.035),
            "brow_w": int(head_w * 0.38 * ident.eye_width),
            "brow_t": max(2, int(head_w * 0.075 * ident.brow_thickness)),
            "nose_w": int(head_w * 0.22 * ident.nose_width),
            "nose_l": int(head_h * 0.26 * ident.nose_length),
            "nose_y": cy + int(head_h * 0.08),
            "mouth_w": int(head_w * 0.40 * ident.mouth_width),
            "mouth_h": int(head_h * 0.09 * ident.mouth_height),
            "mouth_y": cy + int(head_h * 0.33),
            "ear_s": int(head_w * 0.20 * ident.ear_size),
        }

    # ------------------------------------------------------------- primitives

    def _background(self, ap: Appearance, w: int, h: int) -> np.ndarray:
        top = np.array(_hsv(ap.background_hue, 0.22, 0.62 * ap.brightness), dtype=np.float32)
        bottom = np.array(
            _hsv((ap.background_hue + 0.05) % 1.0, 0.30, 0.42 * ap.brightness), dtype=np.float32
        )
        ramp = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        col = top[None, :] * (1 - ramp) + bottom[None, :] * ramp
        canvas = np.repeat(col[:, None, :], w, axis=1)
        canvas = np.clip(canvas, 0, 255)
        return _add_grain(canvas.astype(np.uint8), 2.0)

    def _draw_ink(self, ink, ident: Identity, ap: Appearance, skin, g: dict) -> None:
        """Crisp contour + feature lines. Kept thin so the embedding stays geometric."""
        cx, cy, hw, hh = g["cx"], g["cy"], g["head_w"], g["head_h"]
        lw = max(2, hw // 34)
        ink_colour = self._shade(skin, 0.40)
        cv2.ellipse(ink, (cx, cy), (hw, hh), 0, 0, 360, ink_colour, lw, cv2.LINE_AA)

        # Jaw contour, implied by the taper.
        jaw = int(ident.jaw_taper * hh * 0.45)
        if jaw > 4:
            cv2.ellipse(
                ink, (cx, cy + int(hh * 0.34)), (int(hw * (1 - ident.jaw_taper * 0.42)), jaw),
                0, 15, 165, ink_colour, lw, cv2.LINE_AA,
            )

        for sign in (-1, 1):
            ex = cx + sign * int(g["eye_spacing"])
            cv2.ellipse(
                ink, (ex, g["eye_y"]), (g["eye_w"], g["eye_h"]), 0, 0, 360, ink_colour,
                max(2, lw // 2), cv2.LINE_AA
            )
            # Upper lash line carries the expression.
            cv2.ellipse(
                ink, (ex, g["eye_y"]), (g["eye_w"], g["eye_h"]),
                0, 195, 345, self._shade(skin, 0.26), max(2, lw), cv2.LINE_AA
            )
            bx = ex
            angle = ident.brow_angle * sign + ap.brow_raise * 0.18
            cv2.line(
                ink,
                (bx - g["brow_w"] // 2, g["brow_y"] + int(angle * hh * 0.05)),
                (bx + g["brow_w"] // 2, g["brow_y"] - int(angle * hh * 0.05)),
                ink_colour, g["brow_t"], cv2.LINE_AA,
            )
            # Nose wing.
            cv2.ellipse(
                ink,
                (cx + sign * int(g["nose_w"] * 0.72), g["nose_y"] + int(g["nose_l"] * 0.60)),
                (max(3, g["nose_w"] // 4), max(3, int(g["nose_l"] * 0.14))),
                0, 0, 360, ink_colour, max(2, lw // 2), cv2.LINE_AA,
            )
            # Mouth corner, lifted by expression.
            cv2.ellipse(
                ink, (cx, g["mouth_y"]), (g["mouth_w"], g["mouth_h"]),
                0, 195 + int(ap.smile * 12), 345 + int(ap.smile * 12),
                ink_colour, max(2, lw), cv2.LINE_AA,
            )
            # Ear outline.
            cv2.ellipse(
                ink, (cx + sign * int(hw * 0.97), cy + int(hh * 0.04)),
                (g["ear_s"], int(g["ear_s"] * 1.5)), 0, 0, 360, ink_colour, lw, cv2.LINE_AA
            )

    def _skin_colour(self, tone: float, light_dir: float, brightness: float) -> tuple[int, int, int]:
        """Skin tone is an identity property, so hue and saturation derive from tone
        alone.

        Earlier versions folded ``light_dir`` and ``brightness`` into hue and value,
        which meant a panel of the same character under different lighting produced a
        measurably different face. Face verification models lean heavily on colour, so
        that turned a nuisance variable into noise in the identity measurement.
        Lighting now only shows up in shading geometry.
        """
        hue = int(np.clip(0.055 + tone * 0.045, 0, 179))
        sat = int(np.clip(58 + (1 - tone) * 40, 0, 255))
        val = float(np.clip(0.94 - tone * 0.42, 0, 1))
        return hue, sat, int(val * 255)

    def _shade(self, colour: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
        h, s, v = colour
        return h, s, int(np.clip(v * factor, 0, 255))

    def _draw_head(self, canvas, ident: Identity, skin, ap: Appearance, g: dict) -> None:
        cx, cy, hw, hh = g["cx"], g["cy"], g["head_w"], g["head_h"]

        # Neck first so the head overlaps it, with a cast shadow under the jaw.
        neck_w = int(hw * 0.44)
        cv2.rectangle(canvas, (cx - neck_w, cy + int(hh * 0.40)),
                      (cx + neck_w, cy + int(hh * 1.00)), self._shade(skin, 0.74), -1)
        cv2.ellipse(canvas, (cx, cy + int(hh * 0.44)), (int(neck_w * 1.5), int(hh * 0.16)),
                    0, 0, 360, self._shade(skin, 0.62), -1, cv2.LINE_AA)

        cv2.ellipse(canvas, (cx, cy), (hw, hh), 0, 0, 360, skin, -1, cv2.LINE_AA)

        # Jaw taper: narrow the lower half with an inscribed ellipse, not a hard wedge.
        if ident.jaw_taper > 0.05:
            cv2.ellipse(
                canvas, (cx, cy + int(hh * 0.16)),
                (int(hw * (1.0 - ident.jaw_taper * 0.30)), int(hh * 0.86)),
                0, 0, 180, skin, -1, cv2.LINE_AA,
            )

        chin = int(ident.chin_length * hh * 0.07)
        cv2.ellipse(canvas, (cx, cy + hh - chin), (int(hw * 0.44), int(chin * 1.7)),
                    0, 0, 360, skin, -1, cv2.LINE_AA)

        cheek = int(ident.cheek_fullness * hw * 0.22)
        for sign in (-1, 1):
            cv2.ellipse(canvas, (cx + sign * int(hw * 0.50), cy + int(hh * 0.10)),
                        (cheek, int(cheek * 1.3)), 0, 0, 360, skin, -1, cv2.LINE_AA)

        self._form_shadow(canvas, ident, skin, ap, g)

    def _form_shadow(self, canvas, ident: Identity, skin, ap: Appearance, g: dict) -> None:
        """Brow ridge, eye sockets, cheekbone and nose shadow. This is what makes the
        flat fill read as a face to a detector rather than a circle."""
        cx, cy, hw, hh = g["cx"], g["cy"], g["head_w"], g["head_h"]
        depth = 0.30 + ident.eye_depth * 0.34
        socket = self._shade(skin, 1.0 - depth * 0.34)

        if ident.eye_depth > 0.02:
            for sign in (-1, 1):
                cv2.ellipse(canvas,
                            (cx + sign * int(g["eye_spacing"]), cy - int(hh * 0.11)),
                            (int(hw * 0.34), int(hh * 0.22)),
                            0, 0, 360, socket, -1, cv2.LINE_AA)

        # Cheekbone highlight + shadow beneath it.
        for sign in (-1, 1):
            bx = cx + sign * int(hw * 0.58)
            by = cy + int(hh * 0.06)
            cv2.ellipse(canvas, (bx, by), (int(hw * 0.24), int(hh * 0.16)),
                        0, 0, 360, self._shade(skin, 1.05), -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (bx, by + int(hh * 0.17)), (int(hw * 0.22), int(hh * 0.11)),
                        0, 0, 360, self._shade(skin, 0.88), -1, cv2.LINE_AA)

        # Temples fall off toward the sides of the head.
        for sign in (-1, 1):
            cv2.ellipse(canvas, (cx + sign * int(hw * 0.90), cy - int(hh * 0.06)),
                        (int(hw * 0.18), int(hh * 0.38)),
                        0, 0, 360, self._shade(skin, 0.90), -1, cv2.LINE_AA)

        # Forehead light and nose shadow.
        cv2.ellipse(canvas, (cx, cy - int(hh * 0.42)), (int(hw * 0.56), int(hh * 0.22)),
                    0, 0, 360, self._shade(skin, 1.04), -1, cv2.LINE_AA)
        nose_shadow = self._shade(skin, 0.84)
        cv2.ellipse(canvas, (cx + int(g["nose_w"] * 0.85), g["nose_y"] - int(hh * 0.02)),
                    (int(g["nose_w"] * 0.62), int(g["nose_l"] * 0.52)),
                    0, 0, 360, nose_shadow, -1, cv2.LINE_AA)

        # Directional light from light_dir.
        if abs(ap.light_dir) > 0.12:
            side = 1 if ap.light_dir > 0 else -1
            cv2.ellipse(canvas, (cx - side * int(hw * 0.55), cy),
                        (int(hw * 0.30), int(hh * 0.52)),
                        0, 0, 360, self._shade(skin, 1.06), -1, cv2.LINE_AA)

    def _draw_ears(self, canvas, ident: Identity, skin, g: dict) -> None:
        cx, cy, hw, hh, es = g["cx"], g["cy"], g["head_w"], g["head_h"], g["ear_s"]
        for sign in (-1, 1):
            ex, ey = cx + sign * int(hw * 0.97), cy + int(hh * 0.04)
            cv2.ellipse(canvas, (ex, ey), (es, int(es * 1.5)), 0, 0, 360, skin, -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (ex, ey), (int(es * 0.46), int(es * 0.82)), 0, 0, 360,
                        self._shade(skin, 0.78), 2, cv2.LINE_AA)

    def _draw_eyes(self, canvas, ident: Identity, ap: Appearance, skin, g: dict) -> None:
        cx = g["cx"]
        ew, eh = g["eye_w"], g["eye_h"]
        for sign in (-1, 1):
            ex = cx + sign * int(g["eye_spacing"])
            ey = g["eye_y"]
            cv2.ellipse(canvas, (ex, ey), (ew + 4, eh + 5), 0, 0, 360,
                        self._shade(skin, 0.84), -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (ex, ey), (ew, eh), 0, 0, 360, (242, 242, 246), -1, cv2.LINE_AA)
            iris_r = max(3, int(min(ew, eh) * 0.80))
            iris_x = ex + int(ap.yaw * ew * 0.28)
            cv2.circle(canvas, (iris_x, ey), iris_r,
                       _hsv((ident.hair_hue * 0.85 + 0.36) % 1.0, 0.68, 0.42), -1, cv2.LINE_AA)
            cv2.circle(canvas, (iris_x, ey), max(2, iris_r // 2), (20, 20, 24), -1, cv2.LINE_AA)
            cv2.circle(canvas, (iris_x - iris_r // 3, ey - iris_r // 3),
                       max(1, iris_r // 4), (252, 252, 255), -1, cv2.LINE_AA)

    def _draw_brows(self, canvas, ident: Identity, ap: Appearance, skin, g: dict) -> None:
        colour = self._shade(skin, 0.46)
        for sign in (-1, 1):
            bx = g["cx"] + sign * int(g["eye_spacing"])
            angle = ident.brow_angle * sign + ap.brow_raise * 0.18
            cv2.line(canvas,
                     (bx - g["brow_w"] // 2, g["brow_y"] + int(angle * g["head_h"] * 0.05)),
                     (bx + g["brow_w"] // 2, g["brow_y"] - int(angle * g["head_h"] * 0.05)),
                     colour, g["brow_t"], cv2.LINE_AA)

    def _draw_nose(self, canvas, ident: Identity, ap: Appearance, skin, g: dict) -> None:
        cx, hw = g["cx"], g["head_w"]
        nw, nl, ny = g["nose_w"], g["nose_l"], g["nose_y"]
        cv2.line(canvas, (cx + int(ap.yaw * nw * 0.35), ny - nl),
                 (cx + int(ap.yaw * nw * 0.45), ny + nl // 2),
                 self._shade(skin, 1.07), max(2, nw // 5), cv2.LINE_AA)
        cv2.ellipse(canvas, (cx, ny + nl // 2), (nw, int(nl * 0.32)), 0, 0, 180,
                    self._shade(skin, 0.86), -1, cv2.LINE_AA)
        for sign in (-1, 1):
            cv2.ellipse(canvas,
                        (cx + sign * int(nw * 0.70), ny + nl // 2 + int(nl * 0.10)),
                        (max(2, nw // 5), max(2, int(nl * 0.12))),
                        0, 0, 360, self._shade(skin, 0.58), -1, cv2.LINE_AA)

    def _draw_mouth(self, canvas, ident: Identity, ap: Appearance, g: dict) -> None:
        mw, mh, my = g["mouth_w"], g["mouth_h"], g["mouth_y"]
        lip = _hsv(0.98, 0.52 + ident.lip_fullness * 0.26, 0.52 + ident.lip_fullness * 0.26)
        cv2.ellipse(canvas, (g["cx"], my), (mw, mh), 0, 0, 360, lip, -1, cv2.LINE_AA)
        upper = int(ap.smile * mh * 0.9)
        cv2.ellipse(canvas, (g["cx"], my - upper // 2), (mw, mh), 0, 195, 345,
                    self._shade(lip, 0.58), max(2, mh // 3), cv2.LINE_AA)
        if ap.smile > 0.25:
            cv2.ellipse(canvas, (g["cx"], my + upper // 2), (int(mw * 0.7), int(mh * 0.5)),
                        0, 0, 180, (248, 246, 243), -1, cv2.LINE_AA)

    def _draw_hair(self, canvas, ident: Identity, ap: Appearance, g: dict) -> None:
        cx, cy, hw, hh = g["cx"], g["cy"], g["head_w"], g["head_h"]
        colour = _hsv(ident.hair_hue, 0.60, 0.40)
        style = int(ident.hair_style)
        vol = ident.hair_volume

        if style == 0:
            cv2.ellipse(canvas, (cx, cy - int(hh * 0.28)), (int(hw * 1.03), int(hh * 0.44 * vol)),
                        0, 180, 360, colour, -1, cv2.LINE_AA)
        elif style == 1:
            cv2.ellipse(canvas, (cx, cy - int(hh * 0.24)), (int(hw * 1.07), int(hh * 0.50 * vol)),
                        0, 180, 360, colour, -1, cv2.LINE_AA)
            cv2.fillPoly(canvas, [np.array([
                [cx + int(hw * 0.9), cy - int(hh * 0.44)],
                [cx - int(hw * 1.2), cy - int(hh * 0.64 * vol)],
                [cx - int(hw * 0.5), cy - int(hh * 0.32)],
            ], dtype=np.int32)], colour, cv2.LINE_AA)
        elif style == 2:
            for sign in (-1, 1):
                cv2.ellipse(canvas, (cx + sign * int(hw * 1.00), cy + int(hh * 0.22)),
                            (int(hw * 0.27), int(hh * 0.64 * vol)), 0, 0, 360, colour, -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (cx, cy - int(hh * 0.28)), (int(hw * 1.07), int(hh * 0.46 * vol)),
                        0, 180, 360, colour, -1, cv2.LINE_AA)
        elif style == 3:
            rng = np.random.default_rng(_stable_seed("curls", round(ident.hair_hue, 4)))
            for _ in range(28):
                angle = rng.uniform(0, math.pi)
                r = int(hw * 1.03)
                px = cx + int(math.cos(angle) * r * rng.uniform(0.9, 1.1))
                py = cy - int(hh * 0.22) + int(math.sin(angle) * r * 0.94 * rng.uniform(0.85, 1.05))
                cv2.circle(canvas, (px, py), int(hw * 0.17 * vol), colour, -1, cv2.LINE_AA)
        elif style == 4:
            cv2.ellipse(canvas, (cx, cy - int(hh * 0.32)), (int(hw * 0.99), int(hh * 0.34)),
                        0, 180, 360, self._shade(colour, 1.12), -1, cv2.LINE_AA)
        else:
            cv2.ellipse(canvas, (cx, cy - int(hh * 0.30)), (int(hw * 1.01), int(hh * 0.42 * vol)),
                        0, 180, 360, colour, -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx + int(hw * 0.10), cy - int(hh * 0.64)),
                       int(hw * 0.25), colour, -1, cv2.LINE_AA)

    def _draw_shoulders(self, canvas, ident: Identity, skin, ap: Appearance, g: dict) -> None:
        cx, cy, hw, hh = g["cx"], g["cy"], g["head_w"], g["head_h"]
        # Neutral, deliberately NOT derived from identity: clothing must not carry
        # identity signal, and the face crop excludes it anyway.
        shirt = _hsv(0.58, 0.10, 0.46)
        top = cy + int(hh * 0.94)
        bottom = PANEL_H * SUPERSAMPLE
        cv2.fillPoly(canvas, [np.array([
            [cx - int(hw * 2.1), bottom],
            [cx - int(hw * 1.02), top],
            [cx + int(hw * 1.02), top],
            [cx + int(hw * 2.1), bottom],
        ], dtype=np.int32)], shirt, cv2.LINE_AA)
        lw = max(3, hw // 20)
        cv2.line(canvas, (cx - int(hw * 1.02), top), (cx, top + int(hh * 0.36)),
                 self._shade(shirt, 0.70), lw, cv2.LINE_AA)
        cv2.line(canvas, (cx + int(hw * 1.02), top), (cx, top + int(hh * 0.36)),
                 self._shade(shirt, 0.70), lw, cv2.LINE_AA)
        cv2.line(canvas, (cx - int(hw * 1.02), top), (cx + int(hw * 1.02), top),
                 self._shade(shirt, 1.10), lw, cv2.LINE_AA)

    def _draw_caption(self, image: np.ndarray, caption: str) -> np.ndarray:
        band_h = 96
        band = np.full((band_h, image.shape[1], 3), 248, dtype=np.uint8)
        image = image.copy()
        image[-band_h:, :] = band
        cv2.line(image, (0, image.shape[0] - band_h), (image.shape[1], image.shape[0] - band_h),
                 (30, 30, 30), 3, cv2.LINE_AA)
        cv2.putText(
            image, caption[:64], (24, image.shape[0] - 34),
            cv2.FONT_HERSHEY_DUPLEX, 0.72, (28, 28, 32), 2, cv2.LINE_AA,
        )
        return image


class SyntheticProvider:
    """Provider that generates panels locally with a controlled identity dial.

    Also serves as the baseline ("no identity conditioning") at strength 0.
    """

    name = "synthetic"

    def __init__(self) -> None:
        self.renderer = SyntheticRenderer()
        self.character_cache: dict[str, Identity] = {}

    def character(self, seed: str) -> Identity:
        """Stable identity for a character name. Same name, same face, forever."""
        if seed not in self.character_cache:
            rng = np.random.default_rng(_stable_seed("character", seed))
            self.character_cache[seed] = Identity.random(rng)
        return self.character_cache[seed]

    def appearance(self, seed: str, index: int, strength: float) -> Appearance:
        """Per-panel pose and framing.

        Pose freedom collapses as identity is pinned. This mirrors the real-world
        failure mode of ID-conditioned diffusion models: at high identity weight the
        face locks and every panel becomes the same mugshot.

        Scope of the strength effect, stated precisely because it bounds what the sweep
        can conclude. ``spread`` scales five face fields (yaw, pitch, smile, brow_raise,
        eye_open) and four whole-frame fields (zoom, light_dir, background_hue,
        brightness). So the *background does move with strength*, by construction. The
        control that matters is the one below — the RNG draw — not pixel-level invariance
        of the frame. Consequence for reading results: synthetic identity ground truth is
        unaffected (identity comes from ``blend``), but synthetic *variety* partly
        reflects narrowed framing and lighting rather than only facial sameness, so treat
        the synthetic variety curve as a simulation of the phenomenon, not an independent
        measurement of it.
        """
        # Seeded WITHOUT strength so every panel draws the same pose sample at every
        # setting — same underlying draws, only their spread scaled. Seeding on strength
        # as well meant each condition sampled a different pose, so the sweep measured the
        # seed lottery rather than the identity dial — it flattened AUROC to chance.
        rng = np.random.default_rng(_stable_seed("appearance", seed, index))
        spread = 1.0 - 0.55 * strength
        base_hue = rng.uniform(0, 1)
        return Appearance(
            yaw=float(np.clip(rng.normal(0, 0.42) * spread, -1, 1)),
            pitch=float(np.clip(rng.normal(0, 0.26) * spread, -1, 1)),
            smile=float(np.clip(rng.normal(0, 0.45) * spread, -1, 1)),
            brow_raise=float(np.clip(rng.normal(0, 0.30) * spread, -1, 1)),
            eye_open=float(np.clip(rng.normal(0.92, 0.07) * spread + 0.05 * (1 - spread), 0.25, 1.0)),
            zoom=float(np.clip(rng.normal(1.0, 0.09) * spread + 0.0 * (1 - spread), 0.82, 1.25)),
            light_dir=float(np.clip(rng.normal(0, 0.35) * spread + 0.15 * (1 - spread), -1, 1)),
            background_hue=float((base_hue + rng.uniform(-0.04, 0.04) * spread) % 1.0),
            brightness=float(np.clip(1.0 + rng.normal(0, 0.02) * spread, 0.96, 1.04)),
        )

    def blend(self, character: Identity, seed: str, index: int, strength: float) -> Identity:
        """Mix the character's identity toward a per-panel random draw.

        ``strength`` 1.0 -> pure character. 0.0 -> pure stranger. The interpolation is
        what the identity scorer is being asked to detect.
        """
        rng = np.random.default_rng(_stable_seed("drift", seed, index))
        stranger = Identity.random(rng).as_vector()
        target = character.as_vector()
        mixed = stranger * (1.0 - strength) + target * strength
        return Identity.from_vector(mixed)

    def ground_truth_identity_distance(
        self, character_seed: str, strength: float, panel_index: int
    ) -> float:
        """Exact identity drift for one panel, in normalised parameter units.

        The renderer knows precisely how far it moved the identity vector, so this is the
        ground truth a real provider can never supply. It lets the benchmark check whether
        the measurement tracks the truth, instead of asking anyone to trust a number.
        """
        scale = np.array(
            [IDENTITY_RANGES[f][1] - IDENTITY_RANGES[f][0] for f in IDENTITY_FIELDS], dtype=np.float32
        )
        target = self.character(str(character_seed)).as_vector()
        blended = self.blend(
            self.character(str(character_seed)), str(character_seed), panel_index, strength
        ).as_vector()
        return float(np.linalg.norm((blended - target) / scale))

    def generate(self, *, prompt: str, reference: Sequence[object] | None = None,
                 identity_strength: float = 0.8, seed: int = 0, panel_index: int = 0,
                 caption: str = "", negative_prompt: str | None = None,
                 seed_tag: str = "panel") -> GeneratedPanel:
        """Render one panel.

        ``negative_prompt`` is accepted and ignored: the renderer has no sampler and
        therefore no failure modes to suppress. It is here so the synthetic provider
        satisfies the same protocol as the hosted ones, and the benchmark can call all
        providers through one path.
        """
        char_seed = _character_seed(reference, prompt)
        strength = float(np.clip(identity_strength, 0.0, 1.0))
        identity = self.blend(
            self.character(char_seed), char_seed, panel_index, strength
        )
        appearance = self.appearance(char_seed, panel_index, strength)
        image = self.renderer.render(identity, appearance, caption=caption, with_caption=bool(caption))
        return GeneratedPanel(
            image=image,
            prompt=prompt,
            face_box=self.renderer.face_box(identity, appearance),
            seed=seed or _stable_seed(seed_tag, char_seed, panel_index) % (2**31),
            meta={
                "provider": self.name,
                "identity_strength": strength,
                "character_seed": char_seed,
                "pose_spread": round(1.0 - 0.55 * strength, 3),
            },
        )

    def reference_sheet(self, character_seed: str, n: int = 3) -> list[np.ndarray]:
        """Generate reference portraits for a character at full identity."""
        identity = self.character(character_seed)
        shots = []
        for i in range(n):
            rng = np.random.default_rng(_stable_seed("refshot", character_seed, i))
            ap = Appearance(
                yaw=float(rng.uniform(-0.25, 0.25)),
                pitch=float(rng.uniform(-0.12, 0.12)),
                smile=float(rng.uniform(-0.15, 0.25)),
                brow_raise=0.0,
                eye_open=1.0,
                zoom=1.02,
                light_dir=float(rng.uniform(-0.4, 0.4)),
                background_hue=float(rng.uniform(0.0, 1.0)),
                brightness=1.0,
            )
            shots.append((self.renderer.render(identity, ap), self.renderer.face_box(identity, ap)))
        return shots

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name=self.name,
            supports_identity_strength=True,
            supports_seed=True,
            identity_param="identity_strength",
            local=True,
            provides_face_box=True,
        measurement_domain="synthetic",
        provides_ground_truth=True,
            notes="Deterministic local renderer. Ground truth for the harness.",
        )

def _add_grain(canvas: np.ndarray, amount: float) -> np.ndarray:
    """Additive luminance noise so flat fills do not look like vector art."""
    rng = np.random.default_rng(_stable_seed("grain", canvas.shape[0], canvas.shape[1]))
    noise = rng.normal(0.0, amount, canvas.shape[:2]).astype(np.float32)
    shifted = canvas.astype(np.float32) + noise[:, :, None]
    return np.clip(shifted, 0, 255).astype(np.uint8)


def _layer_alpha(layer: np.ndarray, w: int, h: int) -> np.ndarray:
    """Coverage mask for a paint layer. Pixels painted away from background map to ~0."""
    mask = layer.astype(np.float32).max(axis=2)
    return np.clip(mask / 255.0, 0.0, 1.0)[:, :, None]


def _soften(layer: np.ndarray, sigma: float) -> np.ndarray:
    """Blur so fills bleed into each other the way real skin does."""
    return cv2.GaussianBlur(layer, (0, 0), sigmaX=max(0.6, sigma), sigmaY=max(0.6, sigma),
                            borderType=cv2.BORDER_REFLECT)


def _vignette(image: np.ndarray, amount: float) -> np.ndarray:
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xx - w / 2) / (w / 2)
    dy = (yy - h / 2) / (h / 2)
    falloff = np.clip(1.0 - amount * (dx**2 + dy**2), 0.0, 1.0)[:, :, None]
    return np.clip(image.astype(np.float32) * falloff, 0, 255).astype(np.uint8)
