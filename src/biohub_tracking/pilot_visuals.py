"""Persist small Plotly figures without embedding a second renderer in notebook output."""
from pathlib import Path
import re
from .io import atomic_text


def save_figure(fig,name,root):
    if not re.fullmatch(r'[0-9a-z_]+',name):raise ValueError('Unsafe figure name.')
    payload=fig.to_json()
    if len(payload.encode())>750*1024:raise ValueError('Plot exceeds 750 KiB; reduce points before displaying.')
    directory=Path(root)/'outputs/pilot/figures';directory.mkdir(parents=True,exist_ok=True)
    atomic_text(directory/(name+'.plotly.json'),payload+'\n')
    # One offline JS bundle beside the HTML files, not inside notebook outputs.
    fig.write_html(str(directory/(name+'.html')),include_plotlyjs='directory',full_html=True,auto_open=False)
    return directory/(name+'.html')
