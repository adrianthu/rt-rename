from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import math
from typing import Callable

from .config import get_model_spec
from .constants import MODEL_SYSTEM_PROMPT
from .guidelines import read_guideline
from .inference import GenerationRequest, extract_prediction_and_confidence, generate_response
from .prompts import render_prompt

ProgressCallback = Callable[[str], None]
RowsCallback = Callable[[list[dict[str, object]]], None]


def check_TG263_name(tg263_list: list[str], structure: str) -> str:
    return "pass" if structure in tg263_list else "fail"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _update_row_from_response(
    row: dict[str, object],
    response_text: str,
    nomenclature_names: set[str],
    confidence_override: str | None = None,
    extra_values: dict[str, object] | None = None,
) -> dict[str, object]:
    prediction, confidence = extract_prediction_and_confidence(response_text)
    verify = check_TG263_name(list(nomenclature_names), prediction)
    updated_row = dict(row)
    updated_row.update(
        {
            "TG263 name": prediction,
            "confidence": confidence_override if confidence_override is not None else confidence,
            "verify": verify,
            "accept": verify == "pass",
            "raw output": response_text,
            "timestamp": _timestamp(),
        }
    )
    if extra_values:
        updated_row.update(extra_values)
    return updated_row


def _run_single_inference(
    model_name: str,
    prompt_name: str,
    guideline: str,
    regions: list[str] | None,
    row: dict[str, object],
    nomenclature_list: list[dict[str, str]],
    nomenclature_names: set[str],
) -> dict[str, object]:
    model_spec = get_model_spec(model_name)
    prompt_text = render_prompt(prompt_name, nomenclature_list, str(row["local name"]))
    response = generate_response(
        GenerationRequest(
            model=model_spec,
            prompt=prompt_text,
            system_prompt=MODEL_SYSTEM_PROMPT,
        )
    )
    return _update_row_from_response(row, response.get("response", ""), nomenclature_names)


def _run_uncertain_inference(
    model_name: str,
    prompt_name: str,
    row: dict[str, object],
    nomenclature_list: list[dict[str, str]],
    nomenclature_names: set[str],
) -> dict[str, object]:
    model_spec = get_model_spec(model_name)
    prompt_text = render_prompt(prompt_name, nomenclature_list, str(row["local name"]))
    responses: list[dict[str, str]] = []
    predictions: list[str] = []

    for _ in range(10):
        response = generate_response(
            GenerationRequest(
                model=model_spec,
                prompt=prompt_text,
                system_prompt=MODEL_SYSTEM_PROMPT,
                temperature=1,
                top_p=0.95,
            )
        )
        responses.append(response)
        prediction, _ = extract_prediction_and_confidence(response.get("response", ""))
        predictions.append(prediction)

    counts = {prediction: predictions.count(prediction) for prediction in set(predictions)}
    most_common_prediction = max(counts, key=counts.get)
    selected_response = next(
        response
        for response in responses
        if extract_prediction_and_confidence(response.get("response", ""))[0]
        == most_common_prediction
    )
    probabilities = [count / len(predictions) for count in counts.values()]
    entropy = -sum(probability * math.log2(probability) for probability in probabilities)

    return _update_row_from_response(
        row,
        selected_response.get("response", ""),
        nomenclature_names,
        confidence_override=f"{entropy:.3f}",
        extra_values={
            "entropy": f"{entropy:.3f}",
            "uncertainty_list": predictions,
        },
    )


def rename_structures(
    model: str,
    prompt: str,
    guideline: str,
    regions: list[str] | None,
    structure_dict: list[dict[str, object]],
    progress_callback: ProgressCallback | None = None,
    row_update_callback: RowsCallback | None = None,
    uncertain: bool = False,
) -> list[dict[str, object]]:
    updated_rows = deepcopy(structure_dict or [])
    if not updated_rows:
        return []

    nomenclature_list = read_guideline(regions, guideline, description=False)
    nomenclature_names = {item["name"] for item in nomenclature_list}

    for index, row in enumerate(updated_rows):
        if progress_callback:
            progress_callback(f"Model running {index + 1}/{len(updated_rows)}...")
        try:
            if uncertain:
                updated_rows[index] = _run_uncertain_inference(
                    model_name=model,
                    prompt_name=prompt,
                    row=row,
                    nomenclature_list=nomenclature_list,
                    nomenclature_names=nomenclature_names,
                )
            else:
                updated_rows[index] = _run_single_inference(
                    model_name=model,
                    prompt_name=prompt,
                    guideline=guideline,
                    regions=regions,
                    row=row,
                    nomenclature_list=nomenclature_list,
                    nomenclature_names=nomenclature_names,
                )
        except Exception as exc:
            failed_row = dict(row)
            failed_row.update(
                {
                    "verify": "fail",
                    "accept": False,
                    "comment": f"Inference failed: {exc}",
                    "timestamp": _timestamp(),
                }
            )
            updated_rows[index] = failed_row
        if row_update_callback:
            row_update_callback(updated_rows)

    return updated_rows


def run_model(
    model: str,
    prompt: str,
    guideline: str,
    region: list[str] | None,
    structure_dict: list[dict[str, object]],
    column_defs: list[dict] | None = None,
    gui: bool = True,
    uncertain: bool = False,
) -> list[dict[str, object]]:
    del column_defs, gui
    return rename_structures(
        model=model,
        prompt=prompt,
        guideline=guideline,
        regions=region,
        structure_dict=structure_dict,
        uncertain=uncertain,
    )
