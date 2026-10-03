"""Script -> panel prompt compiler.

Turns a story script into per-panel generation prompts with deterministic seeds. Keeping
this separate from generation means the exact same prompt set is used by the API, the UI
and the benchmark sweep, so a number in the README is reproducible from the same code
path the demo uses.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Literal

Shot = Literal["extreme_wide", "wide", "medium", "close_up", "extreme_close_up"]

# Shot scale drives how much headroom the composition gets.
SHOT_FRAMING: dict[str, str] = {
    "extreme_wide": "extreme wide establishing shot, subject small in frame",
    "wide": "wide shot, full figure visible",
    "medium": "medium shot, waist up",
    "close_up": "close up portrait, head and shoulders fill the frame",
    "extreme_close_up": "extreme close up, face fills the frame",
}

STYLE_DEFAULT = "graphic novel, ink linework, flat colour, halftone shading"

NEGATIVE_BASE = (
    "extra fingers, deformed hands, mutated anatomy, text artifacts, watermark, "
    "signature, low resolution, blur, jpeg artifacts, duplicate face, "
    "multiple faces, inconsistent features"
)

TOKEN_RE = re.compile(r"[^a-z0-9]+")


def stable_seed(*parts: object) -> int:
    joined = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(joined).digest()[:8], "big") % (2**31)


@dataclass
class Character:
    name: str
    description: str = ""
    style: str = STYLE_DEFAULT

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class PanelSpec:
    beat: str
    caption: str = ""
    shot: Shot = "medium"
    pose: str = ""
    expression: str = ""
    setting: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Script:
    title: str
    character: Character
    panels: list[PanelSpec] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "title": self.title,
            "character": self.character.as_dict(),
            "panels": [p.as_dict() for p in self.panels],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Script":
        character = data.get("character") or {}
        return cls(
            title=data.get("title", "Untitled"),
            character=Character(
                name=character.get("name", "Character"),
                description=character.get("description", ""),
                style=character.get("style") or STYLE_DEFAULT,
            ),
            panels=[PanelSpec(**p) for p in data.get("panels", [])],
        )


def compile_prompt(script: Script, index: int) -> str:
    """Build one panel prompt.

    Order matters: subject and appearance first, then framing, then expression and
    setting. Diffusion models weight earlier tokens more heavily, so the identity-bearing
    description has to lead.
    """
    panel = script.panels[index]
    character = script.character
    parts: list[str] = []

    parts.append(character.description.strip() or f"{character.name}, comic character")
    if panel.pose:
        parts.append(panel.pose.strip())
    parts.append(SHOT_FRAMING.get(panel.shot, panel.shot))
    if panel.expression:
        parts.append(f"{panel.expression.strip()} expression")
    if panel.setting:
        parts.append(panel.setting.strip())
    parts.append(character.style.strip() or STYLE_DEFAULT)
    parts.append("consistent character design")

    return ", ".join(p for p in parts if p)


def compile_negative_prompt() -> str:
    return NEGATIVE_BASE


def panel_seed(script: Script, index: int) -> int:
    """Seed per (script, panel) — deliberately NOT per strength.

    Holding the seed fixed across the sweep is what makes the experiment controlled: the
    identity dial becomes the only variable, so a strength that measures better cannot be
    explained by luckier sampling. Hashing the strength in here (as an earlier version did)
    quietly reintroduced that confound and the curve went flat.
    """
    return stable_seed(script.title, script.character.name, index)


def token_slug(text: str) -> str:
    return TOKEN_RE.sub("-", text.lower()).strip("-")[:64]


def default_script() -> Script:
    """The demo story. Also the fixture the UI loads on first paint."""
    return Script(
        title="The Signal Keeper",
        character=Character(
            name="Mara Vess",
            description=(
                "Mara Vess, a young signal keeper with sharp features, dark cropped hair "
                "and a weathered field jacket"
            ),
        ),
        panels=[
            PanelSpec(
                beat="Mara climbs the lighthouse stairwell at dawn",
                caption="Every morning, the same climb.",
                shot="medium",
                pose="three-quarter view, walking upward",
                expression="tired, focused",
                setting="narrow spiral staircase, cold morning light",
            ),
            PanelSpec(
                beat="She reaches the lamp room and checks the crystal",
                caption="Ninety-one days without a message.",
                shot="close_up",
                pose="front view, leaning over the mechanism",
                expression="concerned",
                setting="glass-walled lamp room, brass instrument",
            ),
            PanelSpec(
                beat="A reply finally blinks through the static",
                caption="Then, on the ninety-second day —",
                shot="extreme_close_up",
                pose="front view, eyes wide",
                expression="shocked, hopeful",
                setting="instrument glow, haze",
            ),
            PanelSpec(
                beat="She laughs, relief breaking across her face",
                caption="— someone was still out there.",
                shot="close_up",
                pose="three-quarter view, head tilted back",
                expression="laughing, relieved",
                setting="lamp room, warm light",
            ),
            PanelSpec(
                beat="Mara writes the reply into the logbook",
                caption="Log entry 91. Contact re-established.",
                shot="medium",
                pose="side view, seated, writing",
                expression="calm, resolved",
                setting="wooden desk, logbook, ink pen",
            ),
            PanelSpec(
                beat="She looks out at the sea, lamp behind her",
                caption="The signal holds.",
                shot="wide",
                pose="back three-quarter view",
                expression="quietly content",
                setting="cliff edge, grey sea, lighthouse behind",
            ),
        ],
    )