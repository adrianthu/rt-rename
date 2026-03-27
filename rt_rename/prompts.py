from __future__ import annotations

from pathlib import Path

from .constants import CONFIG_DIR


def _format_guideline_entry(entry: dict[str, str]) -> str:
    description = entry.get("description", "").strip()
    if description:
        return f"        - Name: {entry['name']}, Description: {description}"
    return f"        - Name: {entry['name']}"


def _normalize_structure_input(structure_input: str) -> str:
    return "_".join(structure_input.split())


def render_prompt(
    file_path: str | Path,
    tg263_list: list[dict[str, str]],
    structure_input: str,
    prompt_dir: Path = CONFIG_DIR,
) -> str:
    prompt_path = Path(file_path)
    if not prompt_path.is_file():
        prompt_path = prompt_dir / str(file_path)

    prompt_template = prompt_path.read_text(encoding="utf-8")
    formatted_guidelines = "\n".join(
        _format_guideline_entry(entry) for entry in tg263_list
    )
    return (
        prompt_template.replace("{TG263_list}", formatted_guidelines)
        .replace("{structure_input}", _normalize_structure_input(structure_input))
    )


parse_prompt = render_prompt
parse_prompt_v2 = render_prompt
