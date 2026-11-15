"""Inspect the existing interpreter without creating environments or installing packages."""
import argparse
import importlib
import json
import sys

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--rl',action='store_true')
    args=parser.parse_args()
    names=['numpy','torch','pandas','scipy']+(['gymnasium','stable_baselines3'] if args.rl else [])
    modules={};missing=[]
    for name in names:
        try:
            module=importlib.import_module(name)
            modules[name]=getattr(module,'__version__','available')
        except ImportError:
            missing.append(name)
    if 'torch' not in missing:
        import torch
        modules['cuda_available']=torch.cuda.is_available()
        modules['cuda_version']=torch.version.cuda
        modules['gpu']=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    print(json.dumps(dict(python=sys.version,modules=modules,missing=missing),indent=2))
    if missing:
        print('Install only missing dependencies in this interpreter:',', '.join(missing))
        raise SystemExit(1)
