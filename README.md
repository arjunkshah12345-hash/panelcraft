# PanelCraft

**How much character identity can you buy before you lose the panel?**

PanelCraft is a measurement harness for character consistency in AI comic generation.
Comic models are sold on "consistent characters," but that claim is almost never
measured. PanelCraft renders one script across a sweep of identity-conditioning strength,
scores every panel against reference portraits with a face embedding model, and plots the
result against how varied the panels actually are — so the tradeoff is a curve you can
read instead of a promise you have to trust.

<!-- Results live in results/. The headline numbers below are reproducible with the two
     commands in "Reproducing the numbers". -->

---

## The idea

Every identity-conditioned image model has an identity-strength knob. Turn it up and the
face stops drifting. Turn it down and the panels stop looking like the same comic. Nobody
documents where the useful setting lives, because the answer depends on the model, the
character, and the shot list.

PanelCraft measures that surface:

| | |
|---|---|
| **Identity axis** | FaceNet (InceptionResnetV1, VGGFace2) embedding of each panel's face crop, best-of-K against reference portraits. |
| **Variety axis** | Mean pairwise distance between panels *with each panel's own face masked out* — pose, framing, lighting, set. |
| **Tradeoff** | Identity plotted against variety, dominated points greyed out, Pareto front and a recommended strength. |

Variety is measured outside the face on purpose. If the face counted toward variety, a
model that stamped one mugshot across six panels would score as "varied", which is exactly
the failure the comic is supposed to reveal.

---

## What the numbers actually say

The interesting part is which numbers are trustworthy. The committed run in
[`results/`](results) is one specific sweep; every number below comes from it or from
`validate_scorer.py`.

### The scorer is trustworthy: AUROC 0.9995 on real faces

```
$ python scripts/validate_scorer.py

Identity scorer validation
  dataset          Olivetti faces (400 images, 40 identities)
  detection rate   0.0%
  same person      mean 0.843  p5 0.608  (3600 pairs)
  different person mean 0.102  p95 0.394  (156000 pairs)
  AUROC            0.9995
  threshold        0.5141
  balanced acc     0.9896  (tpr 0.9883 / fpr 0.0092)
```

Same-person pairs average 0.843, different-person pairs 0.102, and the two barely overlap.
That is the instrument, validated on real photographs where the labels come with the data.
The script exits non-zero below AUROC 0.95, so it works as a test.

### The synthetic renderer's identity axis is **not** trustworthy — and the harness says so

The synthetic renderer exists so the project runs with no API key and produces
byte-identical, reproducible output every time. The generator now hashes reference images
by pixel content (not their Python object address), so sweep seeds are stable across
processes and repeated runs. It draws flat vector faces: a face embedding model trained
on photographs cannot reliably read identity off flat vector art — it mostly picks up pose
and rendering style.

PanelCraft detects this at runtime instead of publishing the number anyway:

Because the renderer knows exactly how far it moved each identity, the harness checks the
measurement against that ground truth. The correlation sits at zero and is nowhere near
significant — and because it is measuring noise, it is not stable run to run (p > 0.3), reflecting that FaceNet on vector art measures noise). The measurement does not track known
drift, so:

- the identity axis is flagged untrustworthy,
- **no recommended strength is emitted**, because it would be derived from that axis,
- and the ground-truth curve is reported instead:

```
strength   variety   TRUE identity distance
   0.00     0.723              1.834
   0.25     0.695              1.376
   0.50     0.648              0.917
   0.75     0.625              0.459
   1.00     0.600              0.000
```

Identity collapses exactly as designed, variety degrades by ~17% — the tradeoff is real,
it is just not measurable with a photo-trained embedder on this material.

### Calibration is labelled in-sample

The sweep reports a threshold fitted to the strongest and weakest panels of *that same
sweep*. That is a fit, not a verification result, so it is returned with
`held_out: false` and a caveat string. The held-out number is the Olivetti one above.

---

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Backend on :8000
.venv/bin/uvicorn api.main:app --reload

# Frontend on :5173 (proxies /api to :8000)
cd web && npm install && npm run dev
```

Open <http://localhost:5173>. Pick a provider, drag the identity-strength dial, and render.
The synthetic provider needs no keys and no network.

### Reproducing the numbers

```bash
.venv/bin/python scripts/validate_scorer.py                 # AUROC on real faces
.venv/bin/python scripts/run_benchmark.py --panels 6        # sweep + plot into results/
```

---

## Providers

| Provider | Identity control | Needs a key |
|---|---|---|
| `synthetic` | native dial on a deterministic local renderer | no |
| `pulid` | native PuLID `identity_strength` | `REPLICATE_API_TOKEN` |
| `instantid` | native InstantID `identity_strength` | `REPLICATE_API_TOKEN` |
| `baseline` | prompt-only, no identity conditioning | `REPLICATE_API_TOKEN` |

All four satisfy one protocol, so the benchmark, the API and the UI treat them
identically. Providers declare a `measurement_domain`: the hosted ones produce
photographic faces and are measured normally; the synthetic one is flagged.

To run the interesting comparison, add a token and sweep each provider:

```bash
export REPLICATE_API_TOKEN=...
.venv/bin/python scripts/run_benchmark.py --provider pulid --panels 6
.venv/bin/python scripts/run_benchmark.py --provider baseline --panels 6
```

The baseline is the control: same script, same seeds, identity conditioning off.

---

## API

| Route | Purpose |
|---|---|
| `GET /api/capabilities` | provider catalogue, availability, measurement domain |
| `GET /api/default-script` | the demo script |
| `POST /api/generate` | render one comic and score it against references |
| `POST /api/score` | score arbitrary panels against arbitrary references |
| `POST /api/sweep` | start a background sweep, returns a job id |
| `GET /api/jobs/{id}` | sweep progress |
| `GET /api/results/{id}` | stored result JSON |
| `GET /api/results/{id}/plot` | tradeoff plot PNG |
| `GET /api/health` | backend and detector status |

Interactive docs at `/docs`.

---

## Design decisions worth defending

**Seeds are held fixed across the sweep.** `panel_seed` deliberately excludes identity
strength. An earlier version hashed the strength into the seed, which meant every
condition sampled different noise — the sweep was measuring the seed lottery, and AUROC
sat at chance (0.36). Holding the seed makes the dial the only variable.

**Embedding centering uses a fixed anchor.** Subtracting a common mode removes the shared
style component that dominates raw similarity. The anchor must be identical across
conditions: centering each condition on its own mean centers every condition on its own
answer and erases the comparison.

**Skin tone is an identity property, lighting is not.** Skin colour derives from the tone
parameter alone; lighting only shades. Folding lighting into hue made the same character
look different between panels, and colour-sensitive face models turned that nuisance
variable into measurement noise.

**Variety masks the face.** See above.

**Threshold calibration is reported as a fit.** Labelled `held_out: false` with a caveat,
because a threshold tuned on the panels it judges is not a verification result.

**Jobs live in memory.** A sweep is tens of diffusion calls and takes minutes; a
restart-persistent job queue would be more machinery than this needs. Storing a result
survives a restart, so nothing valuable is lost.

---

## Layout

```
api/
  main.py                  FastAPI routes, background sweep jobs
  services/
    identity.py            FaceNet embeddings, face crops, scoring, calibration
    benchmark.py           sweep, tradeoff curve, Pareto, plot, ground-truth check
    compiler.py            script -> panel prompts, deterministic seeds
    storage.py             flat-file results and images
    providers/             protocol + synthetic + Replicate adapters + registry
scripts/
  validate_scorer.py       held-out scorer validation (exits non-zero if AUROC < 0.95)
  run_benchmark.py         CLI sweep -> results/*.json, *.png, panels/
web/                       Vite + React comic reader, identity meters, tradeoff chart
results/                   committed run artifacts
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the measurement pipeline in detail.

---

## Limitations

- **The synthetic provider cannot measure identity.** Stated plainly rather than hidden;
  the harness detects and reports it.
- **Threshold calibration is in-sample.** The held-out figure is the Olivetti AUROC.
- **MTCNN detects 0% of the synthetic vector faces** and of 64×64 Olivetti portraits; both
  fall back to centre crops, which are recorded per panel as `detected_face: false`. Real
  diffusion output should be detector-resolved, and that path needs a real key to confirm.
- **One character, one script, one style.** The demo story is fixed; a wider claim would
  need a panel of characters across several scripts.
- **Synthetic variety has a known whole-frame confound.** `spread` also changes zoom, light_dir, background_hue, and brightness, not only face geometry; face-excluded variety is the least bad control, but still mixes pose and set.
- **Reproducibility.** Synthetic results are byte-identical across processes once reference images are hashed by pixel content. The seed derivation previously stringified PIL images (including object addresses) — fixed in this commit.
- **Hosted providers are unverified.** The adapters are written against the Replicate
  predictions API but have never run, because no `REPLICATE_API_TOKEN` was available
  during development. Treat `pulid` / `instantid` / `baseline` as untested code, not as
  results.

### Environment

- `PANELCRAFT_DATA_DIR` (optional): directory for hosted provider caches, job outputs, and persisted artifacts. Defaults to a per-user temp directory.
- `REPLICATE_API_TOKEN` (required for `pulid`, `instantid`, and the hosted `baseline`).

## Deployment

The app is intended to run as a Render API + Vercel frontend.

- **Render (API):** see [`render.yaml`](render.yaml). Expects `REPLICATE_API_TOKEN` (for hosted providers) and `PANELCRAFT_DATA_DIR` if you attach a disk. After creation, note the public service URL (e.g. `https://panelcraft-api.onrender.com`). Render's free tier may need enough memory for PyTorch; adjust plan if needed.
- **Vercel (web):** the build uses `npm run build` in `web/`. [`web/vercel.json`](web/vercel.json) proxies all `/api/*` requests to `https://panelcraft-api.onrender.com/api/:path*`. If your Render service name differs, update that destination.
- **Caveats:** hosted models (`pulid`, `instantid`, `baseline`) call Replicate's predictions API; the request shapes in [`api/services/providers/hosted.py`](api/services/providers/hosted.py) are best-effort and unverified without a live token.


## License

MIT.