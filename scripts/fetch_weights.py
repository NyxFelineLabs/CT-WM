"""Fetch pinned release assets with atomic writes and mandatory SHA-256 checks."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def fetch(verify_only=False):
    manifest=json.loads((ROOT/'configs/weights_manifest.json').read_text())
    for relative,record in manifest['files'].items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Asset path escapes repository')
        if path.exists():
            if digest(path)!=record['sha256']:
                raise ValueError(f'{relative}: existing file hash mismatch; preserve it elsewhere before replacing')
            print('Verified',relative)
            continue
        if verify_only:
            raise FileNotFoundError(f'{relative}: missing; run without --verify-only to download')
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name(path.name+'.download')
        try:
            with urllib.request.urlopen(record['url'],timeout=60) as response,temporary.open('wb') as out:
                while chunk:=response.read(1024*1024):
                    out.write(chunk)
            if temporary.stat().st_size!=record['bytes'] or digest(temporary)!=record['sha256']:
                raise ValueError(f'{relative}: downloaded asset failed integrity check')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        print('Downloaded and verified',relative)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify-only',action='store_true')
    fetch(parser.parse_args().verify_only)
