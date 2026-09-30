from pathlib import Path
import sys,json,copy
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from verify_portfolio import check_path,check_notebook,png_check,verify,SECRET
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('path',['data/a.txt','.venv-gpu/a.py','weights/model.pt','../secret.txt','x.zarr/chunk','kaggle.json','artifacts/config.json'])
def test_private_paths_rejected(path):
 with pytest.raises(ValueError):check_path(path)

def test_public_path_allowed():check_path('reports/portfolio_summary.json')
def test_manifest_exact():assert verify()['status']=='passed'
def test_corrupt_png_rejected():
 data=bytearray(next((ROOT/'assets').glob('*.png')).read_bytes());data[55]^=1
 with pytest.raises(ValueError):png_check(bytes(data))
def test_unexecuted_notebook_rejected(tmp_path):
 data=json.loads((ROOT/'notebooks/00_portfolio_overview.ipynb').read_text());next(c for c in data['cells'] if c['cell_type']=='code')['execution_count']=None
 p=tmp_path/'a.ipynb';p.write_text(json.dumps(data))
 with pytest.raises(ValueError):check_notebook(p)
def test_private_key_pattern():assert SECRET.search('-----BEGIN '+'PRIVATE KEY-----')
