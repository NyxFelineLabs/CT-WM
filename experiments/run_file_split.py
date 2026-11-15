"""Train the reference model with split boundaries aligned to CSV files."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from wm_ms3m.core import WMMS3MConfig, load_data, train_and_eval_wm_ms3m

parser = argparse.ArgumentParser()
parser.add_argument('--width', type=int, required=True)
args = parser.parse_args()
cfg = WMMS3MConfig()
cfg.d_model = args.width
cfg.ckpt_dir = str(ROOT / 'artifacts' / f'file_split_width_{args.width}')
out = Path(cfg.ckpt_dir)
out.mkdir(parents=True, exist_ok=True)
manifest = json.loads((ROOT / 'data/manifest.json').read_text())
files = manifest['files']
assert len(files) == 10
total = files[-1]['sample_stop']
train_end = files[6]['sample_stop']
val_end = files[7]['sample_stop']
# The loader takes integer floors; half-sample offsets avoid float roundoff.
cfg.train_ratio = (train_end + 0.5) / total
cfg.val_ratio = (val_end - train_end + 0.5) / total
train, val, test, meta, *_ = load_data(cfg)
assert [len(train), len(val), len(test)] == [train_end, val_end-train_end, total-val_end]
assert files[6]['sample_stop'] == files[7]['sample_start']
assert files[7]['sample_stop'] == files[8]['sample_start']
split = {'train': files[:7], 'validation': files[7:8], 'test': files[8:],
         'counts': meta['splits'], 'window_length': 32}
(out / 'file_split.json').write_text(json.dumps(split, indent=2))
config = {k: getattr(cfg, k) for k in dir(cfg)
          if not k.startswith('_') and not callable(getattr(cfg, k))}
(out / 'effective_config.json').write_text(json.dumps(config, indent=2))
print('File split:', json.dumps({k: [f['file'] for f in split[k]] for k in ('train', 'validation', 'test')}), flush=True)
print('Counts:', meta['splits'], flush=True)
started = time.perf_counter()
try:
    train_and_eval_wm_ms3m(cfg)
except Exception:
    traceback.print_exc()
    raise
finally:
    print('Elapsed seconds:', time.perf_counter() - started, flush=True)
