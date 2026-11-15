"""Select MPPI settings on the validation CSV only."""
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
from planner_runtime import OPT,make_executor,verify
from planner_mppi import GraphMPPI
from system_common import ROOT,load_bundle,sync,write_json,core
from system_prediction import COLUMNS

@torch.no_grad()
def simulate(b,planner,frames,starts,seeds=[42],timed=True):
    executor=planner.executor
    # Environment graph has a separate buffer and no optimized-policy state.
    env_x=torch.zeros(1,32,13,device='cuda')
    env_graph=torch.cuda.CUDAGraph()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):executor(env_x)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(env_graph):env_y=executor(env_x)
    records=[]
    for file,frame in frames:
        for start in starts(len(frame)):
            for seed in seeds:
                core.set_seed(seed)
                window=frame[start-32:start].copy()
                planner.reset(window)
                for step in range(8):
                    sync();t=time.perf_counter()
                    action,diag=planner.decide(window)
                    sync();td=time.perf_counter()
                    env_x.copy_(torch.as_tensor(window[None],device='cuda'));env_graph.replay()
                    y=env_y[0].cpu().numpy().copy()
                    planner.observe(float(y[10]),action)
                    y_next=y.copy();y_next[5]=action
                    window=np.concatenate([window[1:],y_next[None]],0)
                    sync();tc=time.perf_counter()
                    records.append(dict(file=file,target_row=int(start),seed=seed,step=step+1,
                        SINR_db=float(y[9]),SE=float(y[10]),PRB=action,RSRP_dbm=float(y[7]),
                        decision_ms=1000*(td-t),cycle_ms=1000*(tc-t),
                        joint_candidate=diag['joint_candidate'],safe_candidate=diag['safe_candidate']))
    return pd.DataFrame(records)

def episode_summary(df):
    return df.groupby(['file','target_row','seed'])[['SE','PRB','SINR_db','decision_ms','cycle_ms']].mean()

@torch.no_grad()
def hold_reference(executor,frame,start,low,high):
    window=frame[start-32:start].copy()
    action=float(np.clip(window[-1,5],low,high))
    ys=[]
    for _ in range(8):
        y=executor(torch.tensor(window[None],device='cuda'))[0].cpu().numpy()
        ys.append(y.copy());y[5]=action
        window=np.concatenate([window[1:],y[None]],0)
    return np.stack(ys)[:,10].mean(),action

def main():
    b=load_bundle('L32')
    p=sorted((ROOT/'data/raw').glob('*.csv'))[7]
    frame=pd.read_csv(p)[COLUMNS].to_numpy(np.float32)
    starts=lambda n:np.linspace(64,n-8,12,dtype=int)
    results=[]
    for population,sigma in [(16,3.),(32,3.),(32,5.),(64,5.)]:
        start=time.perf_counter()
        planner=GraphMPPI(b,population=population,sigma=sigma)
        construction=time.perf_counter()-start
        # Warm-up outside measurements.
        planner.reset(frame[32:64]);planner.decide(frame[32:64]);sync()
        df=simulate(b,planner,[(p.name,frame)],starts)
        ref=[hold_reference(planner.executor,frame,int(k),planner.low,planner.high) for k in starts(len(frame))]
        ep=episode_summary(df)
        ep['holdSE']=[v[0] for v in ref];ep['holdPRB']=[v[1] for v in ref]
        ep['se_drop']=1-ep.SE/ep.holdSE;ep['prb_save']=1-ep.PRB/ep.holdPRB
        joint=(ep.se_drop<=.01001)&(ep.prb_save>.15)
        item=dict(population=population,sigma=sigma,temperature=.5,
            mean_se_change_percent=float(100*(ep.SE.mean()/ep.holdSE.mean()-1)),
            mean_prb_saving_percent=float(100*(1-ep.PRB.mean()/ep.holdPRB.mean())),
            joint_episode_success_rate=float(joint.mean()),
            mean_decision_ms=float(df.decision_ms.mean()),p95_decision_ms=float(df.decision_ms.quantile(.95)),
            mean_cycle_ms=float(df.cycle_ms.mean()),construction_seconds=construction)
        results.append(item)
        df.to_csv(OPT/f'validation_mppi_p{population}_s{sigma}.csv',index=False)
        print(item,flush=True)
        write_json(OPT/'validation_search.json',results)
        del planner
    # Joint constraints take priority, then fraction of feasible episodes, then latency.
    selected=max(results,key=lambda x:(x['mean_se_change_percent']>=-1 and x['mean_prb_saving_percent']>15,
                    x['mean_se_change_percent']>=-1,x['p95_decision_ms']<20,
                    x['mean_prb_saving_percent'],x['joint_episode_success_rate'],-x['mean_decision_ms']))
    write_json(OPT/'selected_mppi.json',dict(**selected,selection_split='validation_file_8_only',
        hard_se_reference='>=99% of paired initial hold cumulative SE',
        prb_target='<=84.9% of paired hold cumulative PRB',
        world_model_checkpoint='file_split_width_93/best.pt'))
    print('SELECTED',selected,flush=True)

if __name__=='__main__':main()
