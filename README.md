# RT-Rename

RT-Rename standardizes radiotherapy structure names against TG-263 using local or cloud-hosted LLMs. The repository is now organized so the web UI, parsing logic, DICOM helpers, and inference orchestration are separated, which makes it much easier to extend toward future VLM support.

## What Changed

- Core logic now lives in the `rt_rename/` package instead of a single `utils.py` module.
- The Dash UI is isolated in `rt_rename/web.py`.
- `app.py` and `batch_rename.py` remain as thin entry points.
- Parsing, prompt rendering, guideline loading, inference, exports, and DICOM updates are split into focused modules.
- Basic automated tests were added under `tests/`.

## Repository Layout

```text
rt_rename/
  config.py          Model registry loading
  constants.py       Shared paths and defaults
  dicom_utils.py     RTStruct read/write helpers
  exports.py         CSV export helpers
  guidelines.py      TG-263 workbook loading
  inference.py       Local/cloud inference adapters
  parsers.py         CSV, DICOM, and filename parsing
  prompts.py         Prompt template rendering
  rename_service.py  End-to-end rename orchestration
  web.py             Dash app layout and callbacks
app.py               Web entry point
batch_rename.py      CLI batch entry point
utils.py             Backward-compatible shim
tests/               Unit tests for core logic
```

## Installation

### Docker

1. Clone the repository.
2. Start the stack:

```bash
cd docker
docker-compose up -d
```

3. Open the app at `http://localhost:8055`.

If you want cloud inference, create a root `.env` file with:

```properties
OPEN_AI_URL=your_api_url_here
OPEN_AI_API_KEY=your_api_key_here
```

### Local Development

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r docker/requirements.txt
```

3. Run the web app:

```bash
python app.py
```

## Development Workflow

### Run Tests

```bash
python -m unittest discover -s tests
```

### Batch Rename

The batch entry point is now a CLI instead of a hard-coded script.

```bash
python batch_rename.py input.csv output.csv \
  --model "Llama 3.1 | 70B | local" \
  --prompt prompt_latest.txt \
  --guideline TG263 \
  --region Thorax
```

For DICOM RTStruct input you can also export a renamed RTStruct:

```bash
python batch_rename.py input.dcm output.csv \
  --model "Llama 3.1 | 70B | local" \
  --prompt prompt_latest.txt \
  --output-dicom renamed_rtstruct.dcm
```

## Configuration

### Models

`config/models.json` defines the available models shown in the UI. The loader now supports provider-aware metadata, so we can extend the registry later for multimodal or VLM-capable models without rewriting the app architecture.

Current fields:

- `name`
- `parameters`
- `model_str`
- `cloud`

Optional future-facing fields already supported by the loader:

- `provider`
- `modalities`

### Prompts

Prompt templates live in `config/prompt_*.txt` and are rendered by `rt_rename/prompts.py`.

### TG-263 Nomenclature

The TG-263 workbook is stored at `config/TG263_nomenclature.xlsx`.

## VLM Readiness

The codebase is in a better position for future VLM work because:

- inference requests are funneled through a dedicated adapter layer
- cloud requests already support structured content parts
- UI state is less dependent on module-level globals
- pure rename logic is testable without importing the Dash app

The next natural step for VLM support would be extending `rt_rename/inference.py` and the model registry so image-bearing requests can flow through the same orchestration path.
