"""Experimental controls.

The tradeoff curve is only meaningful if the identity dial is the *only* variable that moves
across a sweep. Two earlier defects quietly broke that:

1. Panel seeds were hashed from ``(script, panel, strength)``, so every strength drew a
   different sample. Any apparent difference was then just luckier rendering, and the curve
   flattened.
2. The synthetic renderer's appearance seed included strength too, so backgrounds and
   lighting changed alongside the face — letting a model "win" on variety by changing the
   weather.

These tests pin both down at the behavioural level rather than by reading the code.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from api.services.compiler import compile_prompt, default_script, panel_seed
from api.services.providers.registry import get_provider


@pytest.fixture(scope="module")
def provider():
    return get_provider("synthetic")


def _outside_faces(low, high, pad: int = 6):
    """Boolean mask covering everything the face boxes do not touch."""
    a, b = low.face_box, high.face_box
    height, width = low.image.shape[:2]
    x1 = max(0, min(a[0], b[0]) - pad)
    y1 = max(0, min(a[1], b[1]) - pad)
    x2 = min(width, max(a[2], b[2]) + pad)
    y2 = min(height, max(a[3], b[3]) + pad)
    mask = np.ones((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = False
    return mask


def test_panel_seed_is_deterministic():
    script = default_script()
    assert panel_seed(script, 0) == panel_seed(script, 0)


def test_panel_seed_varies_by_panel_and_title():
    script = default_script()
    assert panel_seed(script, 0) != panel_seed(script, 1)
    assert panel_seed(script, 0) != panel_seed(replace(script, title="Other Story"), 0)


def test_seed_does_not_depend_on_strength(provider):
    """Regression: strength used to be hashed into the seed."""
    low = provider.generate(prompt="p", reference=["keeper"], identity_strength=0.0, seed=777)
    high = provider.generate(prompt="p", reference=["keeper"], identity_strength=1.0, seed=777)
    assert low.seed == high.seed == 777


def test_background_is_not_invariant_to_strength(provider):
    """Documents a known confound rather than asserting it away.

    ``spread`` scales zoom, light_dir, background_hue and brightness as well as the face
    fields, so whole-frame pixels legitimately differ between strengths. That is intentional
    — pose collapse is the phenomenon being simulated — but it means the synthetic *variety*
    curve is partly constructed rather than independently measured. Identity ground truth is
    unaffected, because identity comes from ``blend``.

    This test exists so that anyone who later decides to make framing strength-invariant
    knows the committed results must be regenerated. The genuine control is the seed test
    above, not pixel equality.
    """
    low = provider.generate(prompt="p", reference=["keeper"], identity_strength=0.0, seed=777)
    high = provider.generate(prompt="p", reference=["keeper"], identity_strength=1.0, seed=777)
    mask = _outside_faces(low, high)
    assert mask.any(), "face boxes covered the frame; the control is untestable"
    assert not np.array_equal(low.image[mask], high.image[mask])


def test_appearance_is_reproducible(provider):
    """Same (seed, index, strength) must give the same framing, or runs are not comparable."""
    a = provider.appearance("keeper", 0, 0.5)
    b = provider.appearance("keeper", 0, 0.5)
    assert a == b


def test_appearance_differs_by_panel(provider):
    """Panels must not all draw the same pose, or variety cannot move at all."""
    seen = {provider.appearance("keeper", i, 0.0).yaw for i in range(6)}
    assert len(seen) > 1


def test_strength_actually_moves_the_face(provider):
    """The converse guard: if the dial did nothing, the sweep would be a flat line."""
    low = provider.generate(prompt="p", reference=["keeper"], identity_strength=0.0, seed=777)
    high = provider.generate(prompt="p", reference=["keeper"], identity_strength=1.0, seed=777)
    height, width = low.image.shape[:2]
    a, b = low.face_box, high.face_box
    inside = (
        slice(max(0, min(a[1], b[1])), min(height, max(a[3], b[3]))),
        slice(max(0, min(a[0], b[0])), min(width, max(a[2], b[2]))),
    )
    assert not np.array_equal(low.image[inside], high.image[inside])
    assert low.meta["pose_spread"] != high.meta["pose_spread"]


def test_generation_is_reproducible(provider):
    a = provider.generate(prompt="p", reference=["keeper"], identity_strength=0.4, seed=99)
    b = provider.generate(prompt="p", reference=["keeper"], identity_strength=0.4, seed=99)
    np.testing.assert_array_equal(a.image, b.image)


def test_reference_images_seed_by_pixel_content(provider):
    """Identical reference pixels must produce an identical character.

    Regression guard: the character seed used to be derived from ``str(reference[0])``, and
    ``str(PIL.Image)`` embeds a memory address. Every sweep therefore rendered a different
    cast, so no two runs were comparable. The seed must depend on content only.
    """
    sheet = [item[0] for item in provider.reference_sheet("keeper", n=2)]
    a = provider.generate(prompt="p", reference=[sheet[0]], identity_strength=0.4, seed=7)
    again = provider.generate(prompt="p", reference=[sheet[0]], identity_strength=0.4, seed=7)
    other = provider.generate(prompt="p", reference=[sheet[1]], identity_strength=0.4, seed=7)
    assert a.meta["character_seed"] == again.meta["character_seed"]
    np.testing.assert_array_equal(a.image, again.image)
    assert other.meta["character_seed"] != a.meta["character_seed"]


def test_reference_image_seed_is_independent_of_object_identity(provider):
    """The (image, face_box) tuple form must hash to the same seed as the bare image."""
    sheet = provider.reference_sheet("keeper", n=1)[0]
    bare, boxed = sheet[0], sheet
    from_bare = provider.generate(prompt="p", reference=[bare], identity_strength=0.4, seed=7)
    from_box = provider.generate(prompt="p", reference=[boxed], identity_strength=0.4, seed=7)
    assert from_bare.meta["character_seed"] == from_box.meta["character_seed"]
    assert from_bare.meta["character_seed"].startswith("img:")
    np.testing.assert_array_equal(from_bare.image, from_box.image)


def test_reference_seed_string_is_passed_through(provider):
    seed_reference = provider.generate(
        prompt="p", reference=["keeper"], identity_strength=0.4, seed=7
    )
    assert seed_reference.meta["character_seed"] == "keeper"


def test_prompt_compilation_is_deterministic():
    script = default_script()
    assert compile_prompt(script, 0) == compile_prompt(script, 0)


def test_provider_reports_synthetic_measurement_domain(provider):
    """Face embeddings are photo-trained; reporting a number for flat vector art is a lie."""
    caps = provider.capabilities()
    assert caps.measurement_domain == "synthetic"
    assert caps.provides_ground_truth is True