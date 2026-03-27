from __future__ import annotations

from dataclasses import dataclass

import ollama
from openai import OpenAI

from .config import ModelSpec
from .constants import DEFAULT_OLLAMA_HOST


@dataclass(frozen=True)
class MessagePart:
    type: str
    text: str | None = None
    image_url: str | None = None

    def as_openai_content(self) -> dict:
        if self.type == "text":
            return {"type": "text", "text": self.text or ""}
        if self.type == "image_url":
            return {
                "type": "image_url",
                "image_url": {"url": self.image_url or ""},
            }
        raise ValueError(f"Unsupported message part: {self.type}")


@dataclass(frozen=True)
class GenerationRequest:
    model: ModelSpec
    prompt: str
    system_prompt: str | None = None
    temperature: float = 0
    top_p: float = 0.1
    content_parts: tuple[MessagePart, ...] = ()

    def user_content(self) -> tuple[MessagePart, ...]:
        return (MessagePart(type="text", text=self.prompt), *self.content_parts)


def extract_response_line(response_text: str) -> str:
    if not response_text:
        return ""
    trimmed = response_text.split("</think>")[-1]
    lines = [line.strip() for line in trimmed.splitlines() if line.strip()]
    return lines[-1] if lines else trimmed.strip()


def extract_prediction_and_confidence(response_text: str) -> tuple[str, str]:
    line = extract_response_line(response_text)
    if not line:
        return "", ""
    prediction, separator, remainder = line.partition(",")
    confidence = remainder if separator else ""
    return prediction.strip(), confidence.strip()


def run_llm(
    model: str = "llama3.1:70b-instruct-q4_0",
    prompt: str | None = None,
    system_prompt: str | None = None,
    temperature: float = 0,
    top_p: float = 0.1,
    gui: bool = False,
) -> dict:
    ollama_client = ollama.Client(host=DEFAULT_OLLAMA_HOST)
    options = {
        "seed": 111,
        "temperature": temperature,
        "top_p": top_p,
        "num_ctx": 24000,
    }
    try:
        return ollama_client.generate(
            model=model,
            prompt=prompt,
            system=system_prompt,
            options=options,
        )
    except ollama._types.ResponseError:
        ollama_client.pull(model=model)
        return ollama_client.generate(
            model=model,
            prompt=prompt,
            system=system_prompt,
            options=options,
        )
    except Exception as exc:
        raise RuntimeError(f"Local model call failed for {model}: {exc}") from exc


def run_llm_cloud(
    model: str,
    prompt: str | None = None,
    system_prompt: str | None = None,
    temperature: float = 0,
    top_p: float = 0.9,
    content_parts: tuple[MessagePart, ...] = (),
) -> dict[str, str]:
    base_url = __import__("os").environ.get("OPEN_AI_URL")
    api_key = __import__("os").environ.get("OPEN_AI_API_KEY")
    if not base_url or not api_key:
        raise RuntimeError(
            "Cloud inference requires OPEN_AI_URL and OPEN_AI_API_KEY environment variables."
        )

    client = OpenAI(base_url=base_url, api_key=api_key)
    user_content = [
        MessagePart(type="text", text=prompt or "").as_openai_content(),
        *[part.as_openai_content() for part in content_parts],
    ]
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    response = client.chat.completions.create(
        model=model,
        max_tokens=32000,
        temperature=temperature,
        top_p=top_p,
        messages=messages,
    )
    return {"response": response.choices[0].message.content or ""}


def generate_response(request: GenerationRequest) -> dict[str, str]:
    if request.model.cloud:
        return run_llm_cloud(
            model=request.model.model_str,
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            top_p=request.top_p,
            content_parts=request.content_parts,
        )

    if request.content_parts:
        raise NotImplementedError(
            "Multimodal request parts are only wired for cloud models right now."
        )

    return run_llm(
        model=request.model.model_str,
        prompt=request.prompt,
        system_prompt=request.system_prompt,
        temperature=request.temperature,
        top_p=request.top_p,
    )
