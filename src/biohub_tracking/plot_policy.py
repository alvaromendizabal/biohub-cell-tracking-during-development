from __future__ import annotations

import base64
import copy
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from IPython.display import HTML, display


def _decode_plotly_arrays(value):
    if isinstance(value, dict):
        if "bdata" in value and "dtype" in value:
            arr = np.frombuffer(base64.b64decode(value["bdata"]), dtype=np.dtype(value["dtype"]))
            shape = value.get("shape")
            if shape:
                dims = tuple(int(v.strip()) for v in str(shape).split(",") if v.strip())
                if dims:
                    arr = arr.reshape(dims)
            return arr.tolist()
        return {k: _decode_plotly_arrays(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_plotly_arrays(v) for v in value]
    return value


def figure_from_saved_json(path: Path | str) -> go.Figure:
    import json
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return go.Figure(_decode_plotly_arrays(raw))


def size_figure(
    fig: go.Figure,
    *,
    kind: str,
    n_categories: int | None = None,
    longest_label: int | None = None,
    title: str | None = None,
) -> go.Figure:
    """Apply chart-specific, employer-facing dimensions."""
    kind = str(kind)
    height = 500
    margins = dict(l=75, r=30, t=85, b=65)

    if kind == "image_overlay":
        height = 760
        margins = dict(l=80, r=30, t=85, b=75)
        fig.update_yaxes(scaleanchor="x", scaleratio=1, autorange="reversed")
    elif kind == "time_series":
        height = 420
        margins = dict(l=65, r=25, t=80, b=60)
    elif kind == "histogram":
        height = 430
        margins = dict(l=65, r=25, t=80, b=60)
    elif kind == "small_bar":
        height = 410
        margins = dict(l=75, r=25, t=80, b=65)
    elif kind == "horizontal_rank":
        n = max(1, int(n_categories or 1))
        label_len = max(10, int(longest_label or 10))
        height = min(950, max(480, 155 + 29 * n))
        margins = dict(l=min(410, max(245, 9 * label_len + 95)), r=35, t=90, b=65)
    elif kind == "matrix":
        height = 680
        margins = dict(l=100, r=40, t=90, b=105)

    existing_title = None
    if fig.layout.title and fig.layout.title.text:
        existing_title = fig.layout.title.text
    fig.update_layout(
        autosize=True,
        height=height,
        margin=margins,
        font=dict(size=13),
        hoverlabel=dict(font_size=12),
        title=dict(text=title or existing_title, x=0.02, xanchor="left", font=dict(size=18)),
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def display_figure(fig: go.Figure, *, label: str) -> None:
    """Render a small CDN-backed HTML fragment inline.

    Each figure includes its own Plotly.js CDN script reference so the first
    plot does not depend on a previous renderer warm-up. The large Plotly
    JavaScript bundle is not embedded into the notebook.
    """
    html = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": True, "displaylogo": False, "scrollZoom": False},
    )
    print(f"{label}_DISPLAY_STARTED html_bytes={len(html.encode('utf-8'))}", flush=True)
    display(HTML(html))
    print(f"{label}_DISPLAY_RETURNED", flush=True)

# BEGIN BIOHUB_SELF_CONTAINED_PLOTLY_POLICY_V1
def render_figure_html(fig):
    """Return self-contained Plotly HTML with no external JS dependency.

    The Plotly JavaScript runtime is embedded directly in each output. This
    deliberately makes notebook outputs larger in exchange for deterministic
    offline/SageMaker rendering and zero browser network dependency.
    """
    return fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={
            "responsive": True,
            "displaylogo": False,
        },
        default_width="100%",
    )


def display_figure(fig, label="PLOTLY", **_kwargs):
    """Display a self-contained Plotly figure and return immediately."""
    from IPython.display import HTML, display

    print(f"{label}_DISPLAY_STARTED", flush=True)
    html = render_figure_html(fig)
    display(HTML(html))
    print(
        f"{label}_DISPLAY_RETURNED "
        f"html_bytes={len(html.encode('utf-8'))}",
        flush=True,
    )
    return None
# END BIOHUB_SELF_CONTAINED_PLOTLY_POLICY_V1

# BEGIN BIOHUB_NATIVE_PLOTLY_MIME_POLICY_V2
PLOTLY_MIME_TYPE = "application/vnd.plotly.v1+json"


def figure_mime_bundle(fig, label="PLOTLY"):
    """Return a compact native Plotly/Jupyter MIME bundle.

    JupyterLab/SageMaker renders application/vnd.plotly.v1+json natively.
    Unlike self-contained HTML, this does not embed a multi-megabyte copy of
    plotly.js in every cell output.
    """
    import json

    payload = json.loads(
        fig.to_json(
            validate=False,
            pretty=False,
            remove_uids=False,
        )
    )
    payload["config"] = {
        "responsive": True,
        "displaylogo": False,
    }

    return {
        PLOTLY_MIME_TYPE: payload,
        "text/plain": f"<Plotly Figure {label}>",
    }


def display_figure(fig, label="PLOTLY", **_kwargs):
    """Display Plotly inline via native Jupyter MIME and return immediately."""
    import json
    from IPython.display import display

    print(f"{label}_DISPLAY_STARTED", flush=True)

    bundle = figure_mime_bundle(fig, label=label)
    payload_bytes = len(
        json.dumps(
            bundle[PLOTLY_MIME_TYPE],
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )

    display(bundle, raw=True)

    print(
        f"{label}_DISPLAY_RETURNED "
        f"mime={PLOTLY_MIME_TYPE} "
        f"payload_bytes={payload_bytes}",
        flush=True,
    )
    return None
# END BIOHUB_NATIVE_PLOTLY_MIME_POLICY_V2
