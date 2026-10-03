# Architecture

## The measurement pipeline

```
script.json
    │  compiler.compile_prompt(panel_index)
    ▼
prompt ──────────────┐
                     │  provider.generate(prompt, reference, identity_strength, seed)
reference portraits ─┘
                              │
                              ▼
                         PIL image + optional face box
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
  identity axis                              variety axis
  crop face (provider box,                    mask each panel's own face,
  else MTCNN, else centre)                   downsample to 32×32, whiten,
        │                                     unit-normalise
        ▼                                           │
  FaceNet InceptionResnetV1                     pairwise L2
  512-d embedding                                  │
        │                                           │
        ▼                                           ▼
  center by reference mean                  variety = mean pairwise distance
  (fixed anchor, identical for every condition)
        │
        ▼
  best-of-K cosine to references  ──►  identity_mean, identity_min, std
        │
        ▼
  tradeoff curve + Pareto front + recommendation
```

## Why each piece is shaped the way it is

### Face crops

Three sources, in order: a provider-supplied box, MTCNN, then a centre crop. The centre
crop is a last resort that silently scores something other than a face, so every panel
records which path was used and `IdentityReport` carries a `detection_rate`. A measurement
that does not report its own coverage is not auditable.

### Embedding centering

Raw face similarities on generated art all sit high, because every panel shares a style,
a crop and a lighting rig. Subtracting a common mean removes that shared mode.

The critical constraint: **the anchor must be identical across conditions.** The first
implementation centred each condition on its own mean. Every condition then centres on
its own answer and the comparison collapses — measured AUROC went to 0.36. The anchor is
now the reference set, which is constant for a given character, so the transform is
comparable and applied identically everywhere.

### Best-of-K, not mean-of-K

A character legitimately looks different across reference angles. The question is "does
this panel look like this character at all," not "does it match this exact photo," so
each panel takes its maximum similarity across references.

### Worst panel decides

`IdentityReport.consistent` uses `identity_min`, not the mean. A comic that is on-model in
five panels and off-model in the sixth still reads as a different character.

### Variety excludes the face

The face is identity. If it counted toward variety, a model stamping one mugshot across
six panels would score as maximally varied — precisely the failure a comic has to expose.
Masking each panel's own face crop leaves pose, framing, lighting and set, which is what
makes the comic readable. No landmark model is needed, so this works on generated and
stylised art where a detector frequently returns nothing.

### Fixed seeds

`panel_seed(script, panel_index)` excludes `identity_strength` on purpose. Hashing the
strength in meant each condition drew different noise, so the sweep measured sampling luck
rather than the dial. The synthetic provider seeds its own pose draw the same way:
`_stable_seed("appearance", seed, index)` with no strength term, and `spread` is the only
thing that responds to strength.

### Ground truth

`SyntheticProvider.ground_truth_identity_distance()` reports the exact normalised distance
the renderer moved each identity. A real provider can never supply this, which is what
makes the synthetic one useful — it is a check on the harness.

Every sweep computes a Spearman correlation between known drift and measured similarity.
If that correlation is weak, the run says so and refuses to emit a recommended strength,
because a recommendation derived from an axis already declared invalid is exactly the kind
of number that should never ship.

## Provider protocol

```python
class Provider(Protocol):
    name: str
    def generate(self, *, prompt, reference, identity_strength,
                 seed, panel_index, negative_prompt=None) -> GeneratedPanel: ...
    def capabilities(self) -> ProviderCapabilities: ...
```

`GeneratedPanel` carries `image`, `prompt`, `seed` and an optional `face_box`. The
`synthetic` renderer returns ndarrays and the hosted adapters return PIL images, so
`benchmark.as_pil()` normalises once at every entry point rather than assuming.

`ProviderCapabilities` declares two things that drive honesty rather than dispatch:

- `measurement_domain` — `"photographic"` or `"synthetic"`. Non-photographic providers get
  their identity axis flagged untrustworthy.
- `provides_ground_truth` — enables the harness self-check.

## Storage

Flat files under `api/services/data/`. A demo's results should be readable with `ls` and
diffable in git. `ResultStore.resolve()` confines reads to the image directory and rejects
paths that escape it.

## Jobs

Sweeps run in a daemon thread with progress written to an in-memory registry. A sweep is
tens of diffusion calls over minutes; a persistent queue would be more machinery than the
product needs, and the README says so rather than implying durability. Results themselves
are persisted, so nothing measurable is lost on restart.

## Frontend

Vite + React, no chart library. The tradeoff plot is one hand-rolled SVG scatter with a
dashed path and two marker classes; a charting dependency would be more bytes than the
chart. During dev, Vite proxies `/api` to the backend so the client uses relative paths in
dev and production alike.