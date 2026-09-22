"""Data-free integrity and privacy checks for the allowlisted publication tree."""
from __future__ import annotations
import ast,base64,binascii,hashlib,json,re,struct,zlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
IGNORED={'.git','__pycache__','.pytest_cache','.ipynb_checkpoints','.venv-review'}
FORBIDDEN={'.aws','.kaggle','.env','data','artifacts','outputs','checkpoints','.venv','.venv-gpu'}
BAD_EXT={'.pt','.pth','.ckpt','.safetensors','.pkl','.pickle','.npz','.npy','.parquet','.zip','.gz','.tar'}
SECRET=re.compile(r'(?:AKIA|ASIA)[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')

def require(ok,message):
 if not ok:raise ValueError(message)

def digest(data):return hashlib.sha256(data).hexdigest()

def png_check(data):
 require(data[:8]==b'\x89PNG\r\n\x1a\n','PNG signature')
 pos=8;dims=None;found_end=False;idat=[]
 while pos<len(data):
  require(pos+12<=len(data),'PNG truncated')
  n=struct.unpack('>I',data[pos:pos+4])[0];kind=data[pos+4:pos+8]
  require(n<=20_000_000 and pos+12+n<=len(data),'PNG chunk bound')
  body=data[pos+8:pos+8+n];crc=struct.unpack('>I',data[pos+8+n:pos+12+n])[0]
  require(binascii.crc32(kind+body)&0xffffffff==crc,'PNG CRC')
  if kind==b'IHDR':
   require(n==13,'PNG header');dims=struct.unpack('>II',body[:8]);require(400<=dims[0]<=2400 and 250<=dims[1]<=2000,'PNG dimensions')
  if kind==b'IDAT':idat.append(body)
  pos+=12+n
  if kind==b'IEND':found_end=True;break
 require(dims and idat and found_end and pos==len(data),'PNG structure')
 raw=zlib.decompressobj().decompress(b''.join(idat),50_000_001)
 require(0<len(raw)<=50_000_000,'PNG decoded bounds')
 return list(dims)

def check_notebook(p):
 nb=json.loads(p.read_text());cells=[c for c in nb['cells'] if c.get('cell_type')=='code']
 require(cells,'Notebook has no code '+str(p));counts=[c.get('execution_count') for c in cells]
 require(all(isinstance(x,int) and x>0 for x in counts),'Unexecuted code '+str(p))
 require(counts==sorted(set(counts)),'Invalid execution order '+str(p))
 png=0
 for c in cells:
  source=c.get('source','');source=''.join(source) if isinstance(source,list) else source
  ast.parse(source)
  for o in c.get('outputs',[]):
   require(o.get('output_type')!='error','Notebook error '+str(p))
   if 'image/png' in o.get('data',{}):
    b=o['data']['image/png'];b=''.join(b) if isinstance(b,list) else b
    png_check(base64.b64decode(b,validate=False));png+=1
 require(png>=1,'Missing saved PNG '+str(p))
 return {'file':p.name,'execution_counts':counts,'png_outputs':png}

def check_path(rel):
 p=Path(rel)
 require(not p.is_absolute() and '..' not in p.parts,'Path escape '+rel)
 require(not (set(p.parts)&FORBIDDEN) and p.suffix.lower() not in BAD_EXT,'Private artifact '+rel)
 require(not any(x.endswith(('.zarr','.geff')) or x.startswith('.venv') for x in p.parts),'Private store '+rel)
 require(p.name!='kaggle.json','Credentials file')

def verify(root=ROOT):
 root=Path(root);manifest=json.loads((root/'PUBLICATION_MANIFEST.json').read_text());expected=manifest['files'];actual=set()
 for p in root.rglob('*'):
  rel=p.relative_to(root)
  if any(x in IGNORED for x in rel.parts):continue
  require(not p.is_symlink(),'Symlink '+str(rel))
  if not p.is_file():continue
  name=rel.as_posix()
  if name=='PUBLICATION_MANIFEST.json':continue
  actual.add(name);check_path(name);data=p.read_bytes()
  require(len(data)<=4_000_000,'Oversized file '+name)
  require(name in expected and digest(data)==expected[name],'Unreviewed or modified file '+name)
  if p.suffix=='.png':png_check(data)
  else:
   text=data.decode('utf-8');require(not SECRET.search(text),'Credential-like value in '+name)
   require(not re.search('arn'+r':aws:[^\s]*:\d{12}:',text),'AWS identifier in '+name)
   if p.suffix=='.py':ast.parse(text,filename=name)
 require(actual==set(expected),'Manifest files missing '+str(set(expected)-actual))
 for p in [root/'README.md',*sorted((root/'docs').glob('*.md')),*sorted((root/'notebooks').glob('*.md'))]:
  for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
   if link.startswith(('http:','https:','mailto:','#')):continue
   target=link.split('#')[0];require((p.parent/target).exists(),'Broken documentation link '+str(p)+': '+link)
 nbs=[check_notebook(p) for p in sorted((root/'notebooks').glob('*.ipynb'))]
 require(len(nbs)==7,'Expected seven evidence notebooks')
 summary=json.loads((root/'reports/portfolio_summary.json').read_text())
 require(summary['official_public_score']==0.947 and summary['latest_attempt']['new_fits']==0,'Evidence contract changed')
 return {'status':'passed','files':len(actual)+1,'bytes':sum((root/n).stat().st_size for n in actual),'notebooks':nbs,'release_id':manifest['release_id']}

if __name__=='__main__':print(json.dumps(verify(),indent=2))
