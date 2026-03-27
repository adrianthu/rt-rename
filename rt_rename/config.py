from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from .constants import MODELS_CONFIG_PATH, PROMPT_GLOB, CONFIG_DIR


@dataclass(frozen=True)
class ModelSpec:
    name: str
    parameters: str
    model_str: str
    cloud: bool = False
    provider: str = ""
    modalities: tuple[str, ...] = ("text",)

    @property
    def display_name(self) -> str:
        location = "cloud" if self.cloud else "local"
        return f"{self.name} | {self.parameters} | {location}"

    @classmethod
    def from_dict(cls, payload: dict) -> "ModelSpec":
        cloud = bool(payload.get("cloud", False))
        provider = payload.get("provider") or (
            "openai-compatible" if cloud else "ollama"
        )
        modalities = tuple(payload.get("modalities", ["text"]))
        return cls(
            name=payload["name"],
            parameters=payload["parameters"],
            model_str=payload["model_str"],
            cloud=cloud,
            provider=provider,
            modalities=modalities,
        )


def load_models(models_path: Path = MODELS_CONFIG_PATH) -> list[ModelSpec]:
    with models_path.open("r", encoding="utf-8") as handle:
        raw_models = json.load(handle)
    return [ModelSpec.from_dict(model) for model in raw_models]


def get_models() -> list[str]:
    return [model.display_name for model in load_models()]


def get_model_spec(display_name: str) -> ModelSpec:
    for model in load_models():
        if model.display_name == display_name:
            return model
    raise ValueError(f"Unknown model selection: {display_name}")


def get_model_str(display_name: str) -> tuple[str, bool]:
    model = get_model_spec(display_name)
    return model.model_str, model.cloud


def get_prompts(prompt_dir: Path = CONFIG_DIR) -> list[str]:
    return sorted(path.name for path in prompt_dir.glob(PROMPT_GLOB))
