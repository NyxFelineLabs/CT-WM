"""Validate ten independent traces and prepare windows with file split boundaries."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ['phy_mcs','mac_dl_cqi','mac_dl_ri','mac_dl_pmi','mac_ul_buffer',
           'mac_n_prb','rsrq','rsrp','rssi','dl_sinr','se','dl_bler','delay']

def prepare(window=32, check_hashes=False):
    if window < 1:
        raise ValueError('Window must be positive')
    reference = json.loads((ROOT/'data/manifest.reference.json').read_text())
    paths = sorted((ROOT/'data/raw').glob('*.csv'))
    if len(paths) != 10:
        raise FileNotFoundError(f'Expected 10 independent CSVs in data/raw, found {len(paths)}. See data/README.md.')
    if [p.name for p in paths] != [r['file'] for r in reference['files']]:
        raise ValueError('CSV names do not match the documented 7/1/2 file protocol')
    frames, records, offset = [], [], 0
    for path, expected in zip(paths, reference['files']):
        frame = pd.read_csv(path)
        if frame.columns.tolist() != ['time'] + COLUMNS:
            raise ValueError(f'{path.name}: incorrect column order')
        if not frame.time.is_monotonic_increasing or not frame.time.is_unique:
            raise ValueError(f'{path.name}: timestamps must be strictly increasing')
        values = frame[COLUMNS].to_numpy(np.float32)
        if not np.isfinite(values).all() or len(values) <= max(window,128):
            raise ValueError(f'{path.name}: invalid values or insufficient rows')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if check_hashes and digest != expected['sha256']:
            raise ValueError(f'{path.name}: SHA-256 differs from historical input')
        size = len(values) - window
        records.append(dict(file=path.name,sha256=digest,rows=len(values),samples=size,
                            sample_start=offset,sample_stop=offset+size,split=expected['split']))
        frames.append(values)
        offset += size
    dest = ROOT/'data' if window==32 else ROOT/f'data/L{window}'
    dest.mkdir(parents=True,exist_ok=True)
    x = np.lib.format.open_memmap(dest/'x.npy',mode='w+',dtype=np.float32,shape=(offset,window,13))
    y = np.lib.format.open_memmap(dest/'y.npy',mode='w+',dtype=np.float32,shape=(offset,13))
    for values, record in zip(frames, records):
        selection = slice(record['sample_start'],record['sample_stop'])
        x[selection] = np.lib.stride_tricks.sliding_window_view(values,window,axis=0)[:-1].transpose(0,2,1)
        y[selection] = values[window:]
    x.flush(); y.flush()
    manifest = dict(window_length=window,files=records,csv_files_kept_separate=True,
                    windows_cross_file_boundaries=False,raw_rows=sum(r['rows'] for r in records),
                    counts={split:sum(r['samples'] for r in records if r['split']==split)
                            for split in ['train','val','test']})
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2))
    return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--window',type=int,default=32)
    parser.add_argument('--check-hashes',action='store_true')
    args=parser.parse_args()
    prepare(args.window,args.check_hashes)
