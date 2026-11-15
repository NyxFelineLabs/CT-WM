"""Frozen protocol for CT-WM benchmark; no changes to source world model."""
import hashlib
import json
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from system_common import ROOT,load_bundle,write_json,core
from system_prediction import COLUMNS

OUT=ROOT/'runs/ct_wm'
OUT.mkdir(parents=True,exist_ok=True)
SEEDS=[42,43,44]
STEPS=128
DEVICE='cuda'

def frames(split):
    paths=sorted((ROOT/'data/raw').glob('*.csv'))
    selected={'train':paths[:7],'val':paths[7:8],'test':paths[-2:]}[split]
    return [(p.name,pd.read_csv(p)[COLUMNS].to_numpy(np.float32)) for p in selected]

def starts(frame):return np.linspace(64,len(frame)-STEPS,8,dtype=int)

def reward_numpy(bundle,y,action,previous,ref_se):
    scaled=(y-np.asarray(bundle.meta['y_full_mean']))/np.asarray(bundle.meta['y_full_std'])
    act=(action-bundle.meta['x_mean'][5])/bundle.meta['x_std'][5]
    features=np.array([scaled[9],scaled[10],scaled[11],scaled[12],act,scaled[7]])
    raw=float(features@bundle.reward_fn.weight.detach().cpu().numpy())
    smooth=bundle.cfg.action_smooth_penalty*abs((action-previous)/bundle.meta['x_std'][5])
    penalty=10*max(0.,.99*ref_se-y[10])/max(abs(ref_se),.05)
    return raw-smooth-penalty

def guard(main):
    try:main()
    except Exception:
        with (OUT/'errors_full.log').open('a',encoding='utf-8') as f:traceback.print_exc(file=f)
        raise

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def protocol():
    model=ROOT/'artifacts/file_split_width_93/best.pt'
    sources={p.name:sha(p) for p in sorted((ROOT/'data/raw').glob('*.csv'))}
    return dict(name='CT-WM',control_interface='PRB only; no MCS, CQI offset or HARQ feedback action',
        train_files=list(sources)[:7],validation_file=list(sources)[7],test_files=list(sources)[-2:],
        source_csv_sha256=sources,checkpoint_sha256=sha(model),
        core_sha256=sha(ROOT/'src/wm_ms3m/core.py'),
        steps=128,starts_per_file=8,episode_seeds=SEEDS,horizon=8,execute_first_action_only=True,
        initial_history=64,environment='Same frozen L32 SSM prior-mean dynamics for every policy; no teacher forcing.',
        reward='Original standardized linear reward and smoothness minus 10*relu(.99*paired_hold_SE-SE)/max(abs(hold_SE),.05). Same criterion for new RL and generic planners.',
        rl_steps_each=50000,rl_training_seed=42,rl_observation='Controlled AND paired-hold standardized L32 histories (832 values); same reference information as planners.',
        training_limit='RL trained in a learned surrogate environment, not model-free interaction with physical O-RAN.',
        evaluation_seed_note='Deterministic fixed/heuristic/RL policies repeat for three seeds; not 48 independent trained agents.',
        attribution='System name CT-WM. SSM/latent backbone and base training derive from frezazadeh/agentic-wm_ms3m commit 29e862d9484ee237eecf669247f29e5db76231f1.',
        calibration='RSRP interval-only postprocessing does not alter SE model, reward or control actions.',
        test_policy='Existing test files have been inspected in prior development; this suite is not a fresh blind holdout. No new test-dependent tuning permitted.',
        latency='CPU physical state -> CPU action; transfer, planner and checks included. Report P99 AND observed maximum. Cycle separately includes environment and GPU drain.')
