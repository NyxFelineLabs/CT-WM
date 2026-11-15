"""External experiment adapters; reference forward/loss/planner are unchanged."""
import json
from pathlib import Path
import sys
import types
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from wm_ms3m import core

ARTIFACTS = ROOT / 'artifacts'
OUT = ROOT / 'runs/system_validation'
OUT.mkdir(parents=True,exist_ok=True)
EXPERIMENTS = {
    'L32': ARTIFACTS / 'file_split_width_93',
    'no_z': ARTIFACTS / 'system_no_z',
    'L16': ARTIFACTS / 'system_L16',
    'L64': ARTIFACTS / 'system_L64',
}

class EmptyGaussian(nn.Module):
    def forward(self, x):
        return x.new_empty((x.shape[0], 0))

class NoLatentMS3M(core.WMMS3MModel):
    """Same backbone and decoders, zero latent dimensions and no prior/post params."""
    def __init__(self, *args, **kwargs):
        kwargs['z_dim'] = 0
        super().__init__(*args, **kwargs)
        self.prior_net = EmptyGaussian()
        self.post_net = EmptyGaussian()

def make_model(cfg, meta, no_z=False):
    cls = NoLatentMS3M if no_z else core.WMMS3MModel
    return cls(in_dim=meta['F'], act_dim=1, d_model=cfg.d_model, z_dim=cfg.z_dim,
               L=meta['L'], n_layers=cfg.n_layers, state_dim=cfg.ssm_state_dim,
               dropout=cfg.dropout, dt_init=cfg.hippo_dt_init,
               n_components=cfg.n_components, out_dim=1, full_dim=meta['F'],
               rsrp_index=cfg.rsrp_index, feature_channel_dropout=cfg.feature_channel_dropout,
               gaussian_input_noise=cfg.gaussian_input_noise,
               use_hetero=cfg.use_heteroscedastic).to(cfg.device)

def load_bundle(name):
    folder = EXPERIMENTS[name]
    cfg = core.WMMS3MConfig()
    for k, v in json.loads((folder / 'effective_config.json').read_text()).items():
        setattr(cfg, k, v)
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.data_dir = str(ROOT / 'data' / (f'L{cfg.data_length}' if hasattr(cfg, 'data_length') else ''))
    cfg.ckpt_dir = str(folder)
    if not (folder / 'best.pt').is_file():
        raise FileNotFoundError('Model weights missing. Run: python scripts/fetch_weights.py')
    ckpt = torch.load(folder / 'best.pt', map_location=cfg.device, weights_only=True)
    meta = ckpt['meta']
    model = make_model(cfg, meta, name == 'no_z').eval()
    model.load_state_dict(ckpt['model'])
    scalers = {}
    for key in ('x', 'y', 'y_full'):
        scalers[key + '_scaler'] = core.StandardScaler(np.array(meta[key + '_mean']),
                                                      np.array(meta[key + '_std']))
    reward = core.RewardModule({n: i for i, n in enumerate(cfg.feature_names)}, cfg).to(cfg.device).eval()
    return types.SimpleNamespace(model=model, cfg=cfg, meta=meta, reward_fn=reward, **scalers)

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def sync():
    torch.cuda.synchronize()
