"""PanelCraft API.

Deliberately small surface: capabilities, generate a comic, score one, sweep the dial,
serve stored results. The interesting work lives in ``api/services``; this module is
transport only.

Long-running work (a sweep is tens of diffusion calls) runs in a background thread and is
tracked in memory. That is the right trade for a hackathon demo — a job that survives a
restart would be more machinery than the product needs, and the note is in the README.
"""

from __future__ import annotations

import io
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from .services import identity as identity_mod
from .services.benchmark import BenchmarkResult, build_references, run_sweep, score_panels
from .services.compiler import (
    Character,
    PanelSpec,
    Script,
    compile_negative_prompt,
    compile_prompt,
    panel_seed,
)
from .services.providers.registry import available_providers, get_provider
from .services.storage import ResultStore

app = FastAPI(
    title="PanelCraft",
    version="0.1.0",
    summary="Character identity consistency benchmark for AI comic generation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = ResultStore()


# ------------------------------------------------------------------ request models


class CharacterModel(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    style: str = Field(default="", max_length=400)


class PanelModel(BaseModel):
    beat: str = Field(default="", max_length=600)
    caption: str = Field(default="", max_length=300)
    shot: Literal[
        "extreme_wide", "wide", "medium", "close_up", "extreme_close_up"
    ] = "medium"
    pose: str = Field(default="", max_length=300)
    expression: str = Field(default="", max_length=200)
    setting: str = Field(default="", max_length=400)


class ScriptModel(BaseModel):
    title: str = Field(default="Untitled", max_length=200)
    character: CharacterModel
    panels: list[PanelModel] = Field(default_factory=list, max_length=24)


class GenerateRequest(BaseModel):
    script: ScriptModel
    provider: str = "synthetic"
    identity_strength: float = Field(default=0.8, ge=0.0, le=1.0)


class ScoreRequest(BaseModel):
    reference_images: list[str]
    panel_images: list[str]
    panel_ids: list[str] | None = None
    threshold: float | None = Field(default=None, ge=-1.0, le=1.0)


class SweepRequest(BaseModel):
    script: ScriptModel | None = None
    provider: str = "synthetic"
    strengths: list[float] = Field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    panels_per_condition: int = Field(default=6, ge=2, le=12)
    reference_count: int = Field(default=3, ge=1, le=6)


def to_script(model: ScriptModel) -> Script:
    from .services.compiler import STYLE_DEFAULT

    return Script(
        title=model.title,
        character=Character(
            name=model.character.name,
            description=model.character.description,
            style=model.character.style or STYLE_DEFAULT,
        ),
        panels=[
            PanelSpec(
                beat=p.beat,
                caption=p.caption,
                shot=p.shot,
                pose=p.pose,
                expression=p.expression,
                setting=p.setting,
            )
            for p in model.panels
        ],
    )


# ------------------------------------------------------------------ jobs


@dataclass
class Job:
    id: str
    kind: str
    status: Literal["running", "done", "error"] = "running"
    progress: float = 0.0
    message: str = "starting"
    result_id: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": round(self.progress, 3),
            "message": self.message,
            "result_id": self.result_id,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def _update(job_id: str, **changes: Any) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        for key, value in changes.items():
            setattr(job, key, value)


# ------------------------------------------------------------------ routes


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "scorer_backend": identity_mod.get_scorer().backend,
        "detector": identity_mod.get_scorer().detector_kind,
        "providers": [c["name"] for c in available_providers()],
    }


@app.get("/api/capabilities")
def capabilities() -> list[dict]:
    """Provider catalogue for the UI, with availability and measurement domain resolved."""
    return available_providers()


@app.get("/api/default-script")
def default_script() -> dict:
    from .services.compiler import default_script as build_default

    return build_default().as_dict()


@app.get("/api/compile/{index}")
def compile_one(index: int, title: str = "The Signal Keeper") -> dict:
    """Expose the compiler so the UI can show exactly what will be sent."""
    script = to_script(
        ScriptModel(
            title=title,
            character=CharacterModel(
                name="Mara Vess",
                description=(
                    "Mara Vess, a young signal keeper with sharp features, dark cropped "
                    "hair and a weathered field jacket"
                ),
            ),
            panels=[
                PanelModel(beat="", shot="medium") for _ in range(max(1, index + 1))
            ],
        )
    )
    return {
        "index": index,
        "prompt": compile_prompt(script, index),
        "negative_prompt": compile_negative_prompt(),
        "seed": panel_seed(script, index),
    }


@app.post("/api/generate")
def generate(request: GenerateRequest) -> dict:
    """Render one comic at one identity strength."""
    try:
        provider = get_provider(request.provider)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    script = to_script(request.script)
    if not script.panels:
        raise HTTPException(status_code=400, detail="script needs at least one panel")

    negative = compile_negative_prompt()
    panels: list[dict] = []
    images: list[Any] = []
    face_boxes: list[Any] = []

    for index in range(len(script.panels)):
        generated = provider.generate(
            prompt=compile_prompt(script, index),
            reference=[script.character.name],
            identity_strength=request.identity_strength,
            panel_index=index,
            seed=panel_seed(script, index),
            negative_prompt=negative,
            caption=script.panels[index].caption,
        )
        images.append(getattr(generated, "image", None))
        face_boxes.append(getattr(generated, "face_box", None))
        panels.append(
            {
                "index": index,
                "beat": script.panels[index].beat,
                "caption": script.panels[index].caption,
                "seed": getattr(generated, "seed", None),
                "prompt": getattr(generated, "prompt", None),
                "image_url": store.save_image(
                    f"gen_{uuid.uuid4().hex[:10]}_{index}.png", generated.image
                ),
            }
        )

    # Score the comic immediately. A comic that renders without telling you whether the
    # character held up is half a product — the whole claim is the measurement.
    report = score_panels(provider, script, images, face_boxes)

    return {
        "provider": provider.name,
        "identity_strength": request.identity_strength,
        "script": script.as_dict(),
        "panels": panels,
        "report": report.to_dict(),
    }


@app.post("/api/score")
def score(request: ScoreRequest) -> dict:
    """Score panels against references. Both sides are data URLs or stored paths."""
    import base64

    from .services.benchmark import as_pil

    def decode(item: str):
        if item.startswith("data:"):
            header, _, payload = item.partition(",")
            return as_pil(io.BytesIO(base64.b64decode(payload)))
        return as_pil(store.resolve(item))

    try:
        references = [decode(i) for i in request.reference_images]
        panels = [decode(i) for i in request.panel_images]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not read images: {exc}") from exc

    if not references or not panels:
        raise HTTPException(status_code=400, detail="need at least one reference and one panel")

    report = identity_mod.score_condition(
        identity_mod.get_scorer(),
        references,
        panels,
        request.panel_ids,
        condition="api",
        threshold=(
            request.threshold
            if request.threshold is not None
            else identity_mod.DEFAULT_THRESHOLD
        ),
    )
    return report.to_dict()


@app.post("/api/sweep")
def sweep(request: SweepRequest) -> dict:
    """Kick off a sweep in the background and return a job id."""
    try:
        get_provider(request.provider)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    script = to_script(request.script) if request.script else None
    if script is None:
        from .services.compiler import default_script as build_default

        script = build_default()
    if not script.panels:
        raise HTTPException(status_code=400, detail="script needs at least one panel")

    job = Job(id=uuid.uuid4().hex[:12], kind="sweep")
    _jobs[job.id] = job

    def work() -> None:
        try:
            result: BenchmarkResult = run_sweep(
                get_provider(request.provider),
                script,
                strengths=request.strengths,
                panels_per_condition=request.panels_per_condition,
                reference_count=request.reference_count,
                progress=lambda label, fraction: _update(
                    job.id, progress=max(0.0, min(1.0, fraction)), message=label
                ),
                on_panel=lambda condition, index, image: store.save_image(
                    f"{job.id}_{condition.replace(' ', '_')}_{index}.png", image
                ),
            )
            payload = result.as_dict()
            result_id = store.save_result(payload)
            plot = store.plot(result, result_id)
            payload = store.load(result_id) or payload
            payload["plot_url"] = f"/api/results/{result_id}/plot"
            store.write(result_id, payload)
            _update(
                job.id,
                status="done",
                progress=1.0,
                message="complete",
                result_id=result_id,
                finished_at=time.time(),
            )
        except Exception as exc:
            _update(
                job.id,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                message="failed",
                finished_at=time.time(),
            )
            traceback.print_exc()

    threading.Thread(target=work, daemon=True, name=f"sweep-{job.id}").start()
    return job.to_dict()


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="no such job")
    return job.to_dict()


@app.get("/api/results")
def list_results() -> list[dict]:
    return store.list()


@app.get("/api/results/{result_id}")
def get_result(result_id: str) -> dict:
    data = store.load(result_id)
    if data is None:
        raise HTTPException(status_code=404, detail="no such result")
    return data


@app.get("/api/results/{result_id}/plot")
def result_plot(result_id: str):
    path = store.plot_path(result_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="plot not generated yet")
    return FileResponse(path, media_type="image/png")


@app.get("/api/images/{filename}")
def stored_image(filename: str):
    try:
        path = store.resolve(filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not path.exists():
        raise HTTPException(status_code=404, detail="no such image")
    return FileResponse(path, media_type="image/png")


@app.delete("/api/results/{result_id}")
def delete_result(result_id: str) -> dict:
    return {"deleted": store.delete(result_id)}


@app.get("/")
def root() -> dict:
    return {
        "name": "PanelCraft",
        "docs": "/docs",
        "idea": "How much character identity can you buy before you lose the panel?",
    }