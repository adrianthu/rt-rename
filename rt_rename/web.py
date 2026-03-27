from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile

from dash import Dash, DiskcacheManager, Input, Output, State, dcc, html, no_update, set_props
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import diskcache
import pydicom

from .config import get_models, get_prompts
from .constants import (
    APP_NAME,
    APP_TITLE,
    APP_VERSION,
    CACHE_DIR,
    DEFAULT_GUIDELINE,
    DEFAULT_HOST,
    DEFAULT_MODEL_DISPLAY,
    DEFAULT_PORT,
    STATUS_IDLE,
)
from .dicom_utils import dataset_from_upload_contents, update_dicom
from .parsers import parse_csv, parse_dicom, parse_filenames
from .rename_service import rename_structures

CACHE_DIR.mkdir(exist_ok=True)
cache = diskcache.Cache(str(CACHE_DIR))
background_callback_manager = DiskcacheManager(cache)


COLUMN_DEFS = [
    {"field": "local name"},
    {"field": "TG263 name"},
    {"field": "confidence"},
    {
        "field": "verify",
        "cellStyle": {
            "styleConditions": [
                {
                    "condition": "params.value == 'pass'",
                    "style": {"color": "green"},
                },
                {
                    "condition": "params.value == 'fail'",
                    "style": {"color": "red"},
                },
            ],
        },
    },
    {"field": "accept", "flex": 1, "width": "10%", "editable": True},
    {"field": "comment", "flex": 1, "editable": True},
    {"field": "raw output", "flex": 1, "editable": True},
]


def _prompt_default() -> str | None:
    prompts = get_prompts()
    if not prompts:
        return None
    if "prompt_latest.txt" in prompts:
        return "prompt_latest.txt"
    return prompts[0]


def _model_default() -> str | None:
    models = get_models()
    if DEFAULT_MODEL_DISPLAY in models:
        return DEFAULT_MODEL_DISPLAY
    return models[0] if models else None


def _with_accept_renderer(column_defs: list[dict]) -> list[dict]:
    updated_columns = deepcopy(column_defs)
    for column in updated_columns:
        if column.get("field") == "accept":
            column["cellRenderer"] = "Checkbox"
            column["cellRendererParams"] = {"disabled": False}
    return updated_columns


def _build_grid() -> dag.AgGrid:
    return dag.AgGrid(
        id="main-data-table",
        rowData=[],
        columnDefs=deepcopy(COLUMN_DEFS),
        dashGridOptions={"domLayout": "autoHeight"},
        style={"height": None},
        csvExportParams={"fileName": "rt_rename.csv"},
    )


def _build_layout() -> html.Div:
    return html.Div(
        [
            dcc.Store(id="uploaded-file-store"),
            html.Div(
                className="column",
                children=[
                    html.H3(
                        children=f"RT-Rename v{APP_VERSION}",
                        style={"textAlign": "left", "margin": "1%"},
                    )
                ],
                style={
                    "width": "70%",
                    "display": "inline-block",
                    "verticalAlign": "bottom",
                },
            ),
            html.Div(
                className="column",
                children=[html.P("Status:", id="status-static")],
                style={
                    "width": "10%",
                    "display": "inline-block",
                    "verticalAlign": "bottom",
                    "textAlign": "right",
                },
            ),
            html.Div(
                className="column",
                children=[
                    html.P(
                        STATUS_IDLE,
                        id="status-bar",
                        style={"color": "green", "marginLeft": "10px"},
                    )
                ],
                style={
                    "width": "20%",
                    "display": "inline-block",
                    "verticalAlign": "bottom",
                },
            ),
            html.Hr(),
            html.H4(children="Settings", style={"textAlign": "left", "margin": "1%"}),
            html.Div(
                className="row",
                children=[
                    html.Div(
                        className="column",
                        children=[
                            html.P("Select a nomenclature:", style={"verticalAlign": "top"}),
                            dcc.Dropdown(
                                ["TG263", "TG263_reverse"],
                                DEFAULT_GUIDELINE,
                                multi=False,
                                id="guideline",
                                style={"width": "80%"},
                            ),
                            html.P(
                                "Select which regions to include:",
                                style={"verticalAlign": "top", "marginTop": "10px"},
                            ),
                            dcc.Dropdown(
                                [
                                    "Thorax",
                                    "Head and Neck",
                                    "Abdomen",
                                    "Limb",
                                    "Pelvis",
                                    "Body",
                                    "Limbs",
                                ],
                                ["Thorax", "Head and Neck", "Body"],
                                multi=True,
                                id="regions",
                                style={"width": "80%"},
                            ),
                        ],
                        style={
                            "width": "33%",
                            "display": "inline-block",
                            "verticalAlign": "top",
                        },
                    ),
                    html.Div(
                        className="column",
                        children=[
                            html.P("Remove target volumes?", style={"verticalAlign": "top"}),
                            dcc.Dropdown(
                                ["False", "True"],
                                "False",
                                multi=False,
                                id="TV-filter",
                                style={"width": "80%"},
                            ),
                            html.P(
                                "Structure Set",
                                style={"verticalAlign": "center", "marginTop": "10px"},
                            ),
                            dcc.Upload(
                                id="upload-data",
                                children=html.Div(
                                    [
                                        html.Br(),
                                        "Drag and drop or click to select files",
                                        html.Br(),
                                        html.Br(),
                                    ]
                                ),
                                style={
                                    "width": "80%",
                                    "lineHeight": "20px",
                                    "borderWidth": "1px",
                                    "borderStyle": "dashed",
                                    "borderRadius": "5px",
                                    "textAlign": "center",
                                    "margin": "10px",
                                    "verticalAlign": "center",
                                },
                                multiple=True,
                            ),
                        ],
                        style={
                            "width": "33%",
                            "display": "inline-block",
                            "verticalAlign": "top",
                        },
                    ),
                    html.Div(
                        className="column",
                        children=[
                            html.P("Model:", style={"verticalAlign": "top"}),
                            dcc.Dropdown(
                                get_models(),
                                _model_default(),
                                multi=False,
                                id="model",
                                style={"width": "80%"},
                            ),
                            html.P("Prompt:", style={"verticalAlign": "top", "marginTop": "10px"}),
                            dcc.Dropdown(
                                get_prompts(),
                                _prompt_default(),
                                multi=False,
                                id="prompt",
                                style={"width": "80%"},
                            ),
                        ],
                        style={
                            "width": "33%",
                            "display": "inline-block",
                            "verticalAlign": "top",
                        },
                    ),
                ],
                style={"marginLeft": "1%", "marginRight": "1%"},
            ),
            html.Hr(),
            html.Div(
                className="row",
                children=[
                    html.Div(
                        className="column",
                        children=[
                            html.Button(
                                "Start renaming",
                                id="button-run-model",
                                n_clicks=0,
                                style={
                                    "borderRadius": "5px",
                                    "width": "80%",
                                    "backgroundColor": "#32a852",
                                    "marginTop": "15px",
                                },
                            )
                        ],
                        style={"width": "33%", "display": "inline-block"},
                    ),
                    html.Div(
                        className="column",
                        children=[
                            dcc.Input(
                                placeholder="Enter patient name/ID here:",
                                type="text",
                                id="patient-name",
                                style={
                                    "marginTop": "15px",
                                    "textAlign": "center",
                                    "width": "87%",
                                },
                            )
                        ],
                        style={
                            "width": "33%",
                            "display": "inline-block",
                            "textAlign": "center",
                        },
                    ),
                    html.Div(
                        className="column",
                        children=[
                            html.Button(
                                "Export csv",
                                id="button-export",
                                n_clicks=0,
                                style={
                                    "borderRadius": "5px",
                                    "width": "100%",
                                    "backgroundColor": "#33b0f2",
                                    "marginTop": "15px",
                                    "float": "right",
                                },
                            )
                        ],
                        style={"width": "16.5%", "display": "inline-block"},
                    ),
                    html.Div(
                        className="column",
                        children=[
                            html.Button(
                                "Export RTstruct",
                                id="button-export-dcm",
                                n_clicks=0,
                                style={
                                    "borderRadius": "5px",
                                    "width": "100%",
                                    "backgroundColor": "#33b0f2",
                                    "marginTop": "15px",
                                    "float": "right",
                                },
                            )
                        ],
                        style={"width": "16.5%", "display": "inline-block"},
                    ),
                    dcc.Download(id="download-dataframe-dcm"),
                ],
                style={"marginLeft": "10px", "marginRight": "10px"},
            ),
            html.Div(children=[_build_grid()], style={"margin": "1%"}),
        ],
        style={"margin": "1%"},
    )


def create_app() -> Dash:
    app = Dash(
        name=APP_NAME,
        title=APP_TITLE,
        external_stylesheets=[dbc.themes.UNITED],
        background_callback_manager=background_callback_manager,
    )
    app.layout = _build_layout()

    @app.callback(
        Output("main-data-table", "rowData"),
        Output("status-bar", "children"),
        Output("uploaded-file-store", "data"),
        Input("upload-data", "filename"),
        Input("upload-data", "contents"),
        State("TV-filter", "value"),
        prevent_initial_call=True,
    )
    def update_on_file_load(filenames, contents, filter_tv):
        if not filenames or not contents:
            return no_update, no_update, no_update

        first_filename = filenames[0]
        lower_name = first_filename.lower()
        if lower_name.endswith(".csv"):
            data = parse_csv(contents[0], first_filename)
            status = f"{len(data)} structures loaded from {first_filename}"
            metadata = {"file_type": "csv", "filename": first_filename}
        elif lower_name.endswith(".dcm"):
            data = parse_dicom(contents[0], first_filename, filter_tv)
            status = f"{len(data)} structures loaded from {first_filename}"
            metadata = {
                "file_type": "dicom",
                "filename": first_filename,
                "contents": contents[0],
            }
        else:
            data = parse_filenames(filenames, filter_tv)
            status = f"{len(data)} structures loaded"
            metadata = {"file_type": "filenames", "filenames": filenames}
        return data, status, metadata

    @app.callback(
        Input("button-run-model", "n_clicks"),
        State("guideline", "value"),
        State("regions", "value"),
        State("model", "value"),
        State("prompt", "value"),
        State("main-data-table", "rowData"),
        State("main-data-table", "columnDefs"),
        running=[(Output("button-run-model", "disabled"), True, False)],
        background=True,
        prevent_initial_call=True,
    )
    def update_on_model_run(
        n_clicks,
        guideline,
        regions,
        model,
        prompt,
        row_data,
        column_defs,
    ):
        del n_clicks
        if not row_data:
            set_props("status-bar", {"children": "No structures loaded."})
            return

        set_props("main-data-table", {"columnDefs": _with_accept_renderer(column_defs)})
        updated_rows = rename_structures(
            model=model,
            prompt=prompt,
            guideline=guideline,
            regions=regions,
            structure_dict=row_data,
            progress_callback=lambda message: set_props(
                "status-bar", {"children": message}
            ),
            row_update_callback=lambda rows: set_props(
                "main-data-table", {"rowData": rows}
            ),
            uncertain=False,
        )
        set_props("main-data-table", {"rowData": updated_rows})
        set_props("status-bar", {"children": "Model run finished!"})

    @app.callback(
        Output("main-data-table", "exportDataAsCsv"),
        Output("main-data-table", "csvExportParams"),
        Input("button-export", "n_clicks"),
        State("patient-name", "value"),
        prevent_initial_call=True,
    )
    def export_data_as_csv(n_clicks, patient_name):
        del n_clicks
        base_name = patient_name or "rt_rename"
        return True, {"fileName": f"{base_name}.csv"}

    @app.callback(
        Output("download-dataframe-dcm", "data"),
        Input("button-export-dcm", "n_clicks"),
        State("main-data-table", "rowData"),
        State("uploaded-file-store", "data"),
        prevent_initial_call=True,
    )
    def download_file(n_clicks, row_data, upload_data):
        del n_clicks
        if not upload_data or upload_data.get("file_type") != "dicom":
            return no_update

        dicom_dataset = dataset_from_upload_contents(upload_data["contents"])
        updated_dataset = update_dicom(dicom_dataset, row_data or [])
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".dcm")
        pydicom.dcmwrite(temp.name, updated_dataset)
        temp.close()
        filename = f"{Path(upload_data['filename']).stem}_renamed.dcm"
        return dcc.send_file(path=temp.name, filename=filename)

    @app.callback(
        Input("main-data-table", "cellRendererData"),
        State("main-data-table", "rowData"),
        prevent_initial_call=True,
    )
    def accept_structures(cell_renderer_data, row_data):
        if not cell_renderer_data or not row_data:
            return
        row_index = int(cell_renderer_data["rowId"])
        row_data[row_index]["accept"] = cell_renderer_data["value"]
        set_props("main-data-table", {"rowData": row_data})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True, host=DEFAULT_HOST, port=DEFAULT_PORT)
