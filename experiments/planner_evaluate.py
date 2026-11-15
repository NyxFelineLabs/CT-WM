"""Evaluate frozen MPPI configurations on the original paired test episodes."""
import copy
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
from planner_runtime import OPT,make_executor,verify,CachedKernel
from planner_mppi import GraphMPPI
from planner_validate import simulate,episode_summary
from system_common import ROOT,core,load_bundle,sync,write_json
from system_prediction import COLUMNS
from system_planning import prior_frame

class FastFixedPolicy:
    def __init__(self,b,mode):
        self.executor=make_executor(b);self.low=b.meta['action_bounds']['low'];self.high=b.meta['action_bounds']['high'];self.mode=mode
    def reset(self,window):
        self.initial=float(np.clip(window[-1,5],self.low,self.high))
    def decide(self,window):
        # Under delayed current-action semantics and w_prb=.2 > smooth=.05,
        # the original one-step greedy solution is exactly the lower bound.
        return (self.low if self.mode=='greedy' else self.initial),dict(joint_candidate=False,safe_candidate=True)
    def observe(self,se,action):pass

def summarize(df,reference):
    ep=episode_summary(df)
    base=episode_summary(reference)
    change=ep/base
    joint=(ep.SE>=.99*base.SE)&(ep.PRB<.85*base.PRB)
    # Pair starts, average planner seeds before exploratory bootstrap.
    ep0=ep.groupby(['file','target_row']).mean()
    base0=base.groupby(['file','target_row']).mean()
    rng=np.random.default_rng(20261111)
    idx=rng.integers(0,len(ep0),size=(2000,len(ep0)))
    se_boot=100*(ep0.SE.to_numpy()[idx].mean(1)/base0.SE.to_numpy()[idx].mean(1)-1)
    prb_boot=100*(1-ep0.PRB.to_numpy()[idx].mean(1)/base0.PRB.to_numpy()[idx].mean(1))
    return dict(episodes=len(ep),decisions=len(df),SINR_db=float(ep.SINR_db.mean()),
        SE=float(ep.SE.mean()),PRB=float(ep.PRB.mean()),
        delta_sinr_db=float(ep.SINR_db.mean()-base.SINR_db.mean()),
        se_change_percent=float(100*(ep.SE.mean()/base.SE.mean()-1)),
        prb_saving_percent=float(100*(1-ep.PRB.mean()/base.PRB.mean())),
        joint_episode_success_rate=float(joint.mean()),
        se_constraint_success_rate=float((ep.SE>=.99*base.SE).mean()),
        prb_target_success_rate=float((ep.PRB<.85*base.PRB).mean()),
        decision_mean_ms=float(df.decision_ms.mean()),decision_p50_ms=float(df.decision_ms.median()),
        decision_p95_ms=float(df.decision_ms.quantile(.95)),decision_max_ms=float(df.decision_ms.max()),
        decision_below_20ms_fraction=float((df.decision_ms<20).mean()),
        cycle_mean_ms=float(df.cycle_ms.mean()),cycle_p95_ms=float(df.cycle_ms.quantile(.95)),
        negative_se_fraction=float((df.SE<0).mean()),
        se_change_exploratory95= np.percentile(se_boot,[2.5,97.5]).tolist(),
        prb_saving_exploratory95=np.percentile(prb_boot,[2.5,97.5]).tolist(),
        aggregate_joint_target_met=bool(ep.SE.mean()>=.99*base.SE.mean() and ep.PRB.mean()<.85*base.PRB.mean()))

@torch.no_grad()
def benchmark_legacy(b,frames):
    # Attribute algorithmic versus deployment-execution improvements explicitly.
    cached=copy.copy(b)
    cached.model=make_executor(b).model
    windows=[v[k-32:k] for _,v in frames for k in np.linspace(64,len(v)-8,4,dtype=int)]
    result=[]
    for name,bundle in [('original_cem_uncached',b),('original_cem_cached_kernels',cached)]:
        for _ in range(2):core.wm_ms3m_cem_plan_next_prb(bundle,windows[0])
        timings=[]
        for seed in [42,43,44]:
            for w in windows:
                core.set_seed(seed)
                sync();t=time.perf_counter();core.wm_ms3m_cem_plan_next_prb(bundle,w);sync()
                timings.append(1000*(time.perf_counter()-t))
        result.append(dict(method=name,repeats=len(timings),mean_ms=float(np.mean(timings)),
                           p95_ms=float(np.percentile(timings,95)),raw_ms=timings))
        print('BENCHMARK',result[-1],flush=True)
    return result

def main():
    b=load_bundle('L32')
    paths=sorted((ROOT/'data/raw').glob('*.csv'))[-2:]
    frames=[(p.name,pd.read_csv(p)[COLUMNS].to_numpy(np.float32)) for p in paths]
    starts=lambda n:np.linspace(64,n-8,8,dtype=int)
    windows=[v[k-32:k] for _,v in frames for k in starts(len(v))]
    verification=verify(b,make_executor(b),windows)
    selected=json.loads((OPT/'selected_mppi.json').read_text())
    results={};rows=[]
    for method in ['hold_cached','greedy_closed_form','mppi_no_se_penalty','mppi_hard_se']:
        build=time.perf_counter()
        if method.startswith('mppi'):
            planner=GraphMPPI(b,population=selected['population'],sigma=selected['sigma'],
                temperature=selected['temperature'],enforce_se=method=='mppi_hard_se')
        else:planner=FastFixedPolicy(b,'greedy' if method=='greedy_closed_form' else 'hold')
        construction=time.perf_counter()-build
        # Warm only deployment buffers; every episode is reset afterward.
        planner.reset(windows[0]);planner.decide(windows[0]);sync()
        df=simulate(b,planner,frames,starts,seeds=[42,43,44])
        df['method']=method;df.to_csv(OPT/f'test_{method}.csv',index=False)
        rows.append(df)
        if method=='hold_cached':reference=df
        item=summarize(df,reference)
        item['construction_seconds']=construction
        results[method]=item
        print(method,item,flush=True)
        write_json(OPT/'planning.json',dict(selected=selected,verification=verification,methods=results))
        del planner
    # Prior original CEM / greedy policies are kept as measured reference traces.
    old=pd.read_csv(ROOT/'runs/system_validation/planning_L32.csv')
    for policy,method in [('cem','original_cem_archived'),('greedy','original_greedy_archived')]:
        df=old[old.policy==policy].copy()
        results[method]=summarize(df,reference)
        results[method]['source']='Archived identical starts/seeds, unchanged weights, prior measured execution.'
    # Common environment and fast-greedy equivalence to archived outputs.
    key=['file','target_row','seed','step']
    current=pd.concat(rows,ignore_index=True)
    ref_archive=old[old.policy=='hold'].set_index(key).sort_index()
    ref_new=reference.set_index(key).sort_index()
    max_env_error=float(np.max(abs(ref_archive[['SE','SINR_db']].to_numpy()-ref_new[['SE','SINR_db']].to_numpy())))
    assert max_env_error<1e-4
    greedy_old=old[old.policy=='greedy'].set_index(key).sort_index()
    greedy_new=current[current.method=='greedy_closed_form'].set_index(key).sort_index()
    assert np.max(abs(greedy_old.PRB.to_numpy()-greedy_new.PRB.to_numpy()))<1e-6
    assert np.max(abs(greedy_old.SE.to_numpy()-greedy_new.SE.to_numpy()))<1e-4
    benchmark=benchmark_legacy(b,frames)
    write_json(OPT/'planning.json',dict(selected=selected,verification=verification,methods=results,
        legacy_latency_recheck=benchmark,common_environment_max_error_vs_archive=max_env_error,
        all_test_starts=16,seeds=[42,43,44],control_steps=8,
        world_model_sha256=hashlib.sha256((ROOT/'artifacts/file_split_width_93/best.pt').read_bytes()).hexdigest(),
        no_test_tuning=True,scope='Same frozen world model only; no real radio network or independent simulator.'))
    pd.DataFrame([dict(method=k,**v) for k,v in results.items()]).to_csv(OPT/'planning_summary.csv',index=False)
    current.to_csv(OPT/'test_steps.csv',index=False)

if __name__=='__main__':main()
