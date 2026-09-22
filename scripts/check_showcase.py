"""Presentation-only integrity checks; no ML code or data execution."""
from pathlib import Path
import json,hashlib,struct,re
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'PUBLICATION_MANIFEST.json').read_text())
expected=set(manifest['files'])|{'PUBLICATION_MANIFEST.json'}
actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts}
assert actual==expected,(actual-expected,expected-actual)
for n,h in manifest['files'].items():
 p=root/n;assert not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==h,n
 assert not any(v in Path(n).parts for v in ['src','research','data','models','weights','.venv']),n
 if p.suffix=='.py':assert n=='scripts/check_showcase.py',n
 if p.suffix=='.ipynb':
  nb=json.loads(p.read_text());assert all(c['cell_type']=='markdown' for c in nb['cells']),n
 if p.suffix=='.png':
  b=p.read_bytes();assert b[:8]==b'\x89PNG\r\n\x1a\n';w,h=struct.unpack('>II',b[16:24]);assert w>=600 and h>=300
 if p.suffix=='.md':
  text=p.read_text();assert '```bash' not in text and '```python' not in text,n
  for target in re.findall(r'\]\(([^)]+)\)',text):
   if '://' in target or target.startswith('#'):continue
   assert (p.parent/target.split('#')[0]).exists(),(n,target)
assert '0.947' in (root/'README.md').read_text()
print('PUBLIC_SHOWCASE_INTEGRITY_PASSED')
