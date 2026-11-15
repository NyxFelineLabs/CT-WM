"""Independent-file L ablations and trained no-latent controlled baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from system_common import ROOT, ARTIFACTS, core, NoLatentMS3M, write_json

COLUMNS = ['phy_mcs', 'mac_dl_cqi', 'mac_dl_ri', 'mac_dl_pmi', 'mac_ul_buffer',
           'mac_n_prb', 'rsrq', 'rsrp', 'rssi', 'dl_sinr', 'se', 'dl_bler', 'delay']
parser = argparse.ArgumentParser()
parser.add_argument('--name', choices=['no_z', 'L16', 'L64'], required=True)
args = parser.parse_args()
length = {'no_z': 32, 'L16': 16, 'L64': 64}[args.name]
cfg = core.WMMS3MConfig()
for key, value in json.loads((ARTIFACTS / 'file_split_width_93/effective_config.json').read_text()).items():
    setattr(cfg, key, value)
cfg.ckpt_dir = str(ARTIFACTS / ('system_' + args.name))
folder = Path(cfg.ckpt_dir)
folder.mkdir(parents=True,exist_ok=True)
if args.name != 'no_z':
    data_dir = ROOT / f'data/L{length}'
    data_dir.mkdir(parents=True,exist_ok=True)
    frames, records, offset = [], [], 0
    for p in sorted((ROOT / 'data/raw').glob('*.csv')):
        df = pd.read_csv(p)
        v = df[COLUMNS].to_numpy(dtype=np.float32)
        assert np.isfinite(v).all() and df.time.is_monotonic_increasing and df.time.is_unique
        count = len(v) - length
        records.append(dict(file=p.name, rows=len(v), sample_start=offset,
                            sample_stop=offset + count, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        frames.append(v)
        offset += count
    x = np.lib.format.open_memmap(data_dir / 'x.npy', mode='w+', dtype=np.float32,
                                 shape=(offset, length, 13))
    y = np.lib.format.open_memmap(data_dir / 'y.npy', mode='w+', dtype=np.float32, shape=(offset, 13))
    for v, r in zip(frames, records):
        s = slice(r['sample_start'], r['sample_stop'])
        x[s] = np.lib.stride_tricks.sliding_window_view(v, length, axis=0)[:-1].transpose(0, 2, 1)
        y[s] = v[length:]
    x.flush(); y.flush()
    cfg.data_dir = str(data_dir)
    train_end, val_end = records[6]['sample_stop'], records[7]['sample_stop']
    cfg.train_ratio = (train_end + 0.5) / offset
    cfg.val_ratio = (val_end - train_end + 0.5) / offset
    write_json(data_dir / 'manifest.json', dict(window_length=length, files=records,
                                              no_shared_raw_rows=True, files_kept_separate=True))
if args.name == 'no_z':
    cfg.z_dim = 0
    core.WMMS3MModel = NoLatentMS3M
train, val, test, meta, *_ = core.load_data(cfg)
assert len(test) == sum(len(pd.read_csv(p)) - length for p in sorted((ROOT / 'data/raw').glob('*.csv'))[-2:])
write_json(folder / 'effective_config.json', {k: getattr(cfg, k) for k in dir(cfg)
           if not k.startswith('_') and not callable(getattr(cfg, k))})
write_json(folder / 'experiment.json', dict(name=args.name, length=length, splits=meta['splits'],
           no_latent=args.name == 'no_z', seed=cfg.seed,
           note='Reference training function unchanged; no-z removes latent pathway only.'))
print(args.name, meta['splits'], flush=True)
start = time.perf_counter()
bundle = core.train_and_eval_wm_ms3m(cfg)
write_json(folder / 'training_run.json', dict(elapsed_seconds=time.perf_counter()-start,
           parameters=sum(p.numel() for p in bundle.model.parameters())))
