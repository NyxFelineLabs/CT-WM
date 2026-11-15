"""One real-window frozen forecast and MPPI decision, without training."""
from pathlib import Path
import json
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
sys.path.insert(0,str(ROOT/'src'))
from e2_sim import PythonLink,VirtualE2Node,frames,OUT
from e2_xapps import Forecast,MPPIXApp
from e2_experiment import exchange
from planner_mppi import load_optimized_planner

if __name__=='__main__':
    if not torch.cuda.is_available():
        raise SystemExit('This smoke test requires CUDA; use scripts/demo.py for the CPU interface demo.')
    torch.set_num_threads(1);torch.manual_seed(42)
    data=frames('test')
    if len(data)!=2:
        raise FileNotFoundError('Two test CSVs are required; see data/README.md')
    _,frame=data[0]
    config=json.loads((OUT/'planner_config.json').read_text())
    link=PythonLink();node=VirtualE2Node(link);forecast=Forecast()
    app=MPPIXApp(link,forecast,config)
    app.reset(node.reset(frame,64,42));app.seed(42)
    row,_,action=exchange(node,app)
    assert 0<=action.mcs_index<=28 and 13<=row['PRB']<=25
    short=load_optimized_planner();short.reset(frame[32:64])
    prb,diag=short.decide(frame[32:64])
    assert 13<=prb<=25
    print(json.dumps(dict(status='PASS',joint_prb=row['PRB'],joint_mcs=row['MCS'],
        joint_decision_ms=row['decision_ms'],short_horizon_prb=prb,
        short_horizon_diagnostics=diag,not_a_latency_benchmark=True),indent=2))
