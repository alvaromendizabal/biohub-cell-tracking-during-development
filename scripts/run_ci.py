"""Bounded synthetic tests; never load competition data or contact Kaggle/AWS."""
from pathlib import Path
import os,subprocess,sys,json,time
from verify_portfolio import verify
ROOT=Path(__file__).resolve().parents[1]

def main():
 start=time.monotonic();print(json.dumps(verify()),flush=True)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(ROOT/'src')+os.pathsep+str(ROOT/'tests'))
 commands=[([sys.executable,'-m','pytest','-q','tests'],ROOT)]
 for d in sorted((ROOT/'research').iterdir()):
  for t in sorted(d.glob('test_*.py')):commands.append(([sys.executable,str(t)],d))
 for cmd,cwd in commands:
  print('SYNTHETIC_TEST',Path(cmd[-1]).name,flush=True)
  subprocess.run(cmd,cwd=cwd,env=env,check=True,timeout=120)
 print(json.dumps({'status':'passed','suite_commands':len(commands),'elapsed_seconds':round(time.monotonic()-start,3),'project_data_fits':0,'network_or_cloud_actions':0}),flush=True)

if __name__=='__main__':main()
