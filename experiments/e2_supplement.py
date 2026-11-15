"""Frozen-policy supplements: paired ablations, greedy + OLLA, SNR shift.

No training entry point. No world-model/core/controller implementation changes.
"""
import argparse
import json
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import SAC,PPO
from e2_sim import (OUT as PREVIOUS,BASE,ROOT,PythonLink,VirtualE2Node,XAppContext,
    E2SMKPMIndication as KPM,frames,starts,write_json,sha)
from e2_xapps import Forecast,OLLAXApp,MPPIXApp,RLXApp,observation
from e2_experiment import exchange

OUT=PREVIOUS/'supplement'
OUT.mkdir(parents=True,exist_ok=True)
MAIN=['olla','prb','greedy_olla','sac','ppo','ctwm','no_safety','no_calib']
ROBUST=['olla','prb','sac','ppo','ctwm']
SNRS=[0,5,10,15,20]
FROZEN=[ROOT/'src/wm_ms3m/core.py',ROOT/'artifacts/file_split_width_93/best.pt',
    PREVIOUS/'rl/sac.zip',PREVIOUS/'rl/ppo.zip',PREVIOUS/'planner_config.json',
    ROOT/'experiments/e2_sim.py',ROOT/'experiments/e2_xapps.py']

def frozen_hashes():return {str(p.relative_to(ROOT)):sha(p) for p in FROZEN}

class MeanOnlyForecast:
    """Requested compound ablation: no dual temperatures and no risk paths.

    Retains frozen validation channel mean correction; removes eight-prior /
    residual stress distribution from safety checks by repeating its mean.
    This cannot isolate a causal effect of RSRP temperature calibration.
    """
    def __init__(self,forecast):self.forecast=forecast
    def predict(self,w,hold,config,cache=False):
        paths,intervals,cost=self.forecast.predict(w,hold,config,cache)
        center=(intervals[0]+intervals[1])/2
        h50=(intervals[1]-intervals[0])/2/np.sqrt(BASE['T50'])
        h95=(intervals[3]-intervals[2])/2/np.sqrt(BASE['T95'])
        unscaled=np.array([center-h50,center+h50,center-h95,center+h95])
        return np.broadcast_to(paths[1:].mean(0),(9,8)).copy(),unscaled,cost

class GreedyOLLAXApp(XAppContext):
    """Fixed, un-tuned rule: channel >15 => -1 PRB; <10 => +1; else hold.

    MCS uses standard own-feedback OLLA at the candidate allocation's effective
    measured SINR. No model, future samples, or reference ACKs are available.
    """
    def decide(self,kpm):
        self.receive(kpm);ch=self.channel()
        dp=-1 if ch>15. else (1 if ch<10. else 0)
        p=int(np.clip(self.current.current_prb+dp,13,25))
        measured=float(self.link.effective(ch,p,self.hold))
        m=self.offset.select_mcs(measured,self.link.threshold10)
        return self.control(p,m),dict(reason='greedy_olla',fallback=False,timeout=False)

class ScenarioNode(VirtualE2Node):
    """New SNR conditions initialize the counterfactual history consistently."""
    def reset(self,frame,start,seed):
        bootstrap=super().reset(frame,start,seed);new=[]
        for wire in bootstrap:
            k=KPM.read(wire);r=np.array(k.kpis)
            channel=r[9]-10*np.log10(self.hold/r[5])
            se,bl=self.link.se(channel,int(r[5]),int(r[0]),self.hold)
            r[10]=float(se);r[11]=float(bl)
            new.append(self.indication(r,k.sequence).wire())
        self.bootstrap=new;self.last_indication=KPM.read(new[-1]);self.current=np.array(self.last_indication.kpis)
        return new

def make_apps(link,forecast,config):
    return dict(olla=OLLAXApp(link),prb=MPPIXApp(link,forecast,config,joint=False,safe=False),
        greedy_olla=GreedyOLLAXApp(link),sac=RLXApp(link,SAC.load(PREVIOUS/'rl/sac',device='cpu')),
        ppo=RLXApp(link,PPO.load(PREVIOUS/'rl/ppo',device='cpu')),ctwm=MPPIXApp(link,forecast,config),
        no_safety=MPPIXApp(link,forecast,config,safe=False),
        no_calib=MPPIXApp(link,MeanOnlyForecast(forecast),config))

def scenarios():
    data=frames('test');channels=np.concatenate([f[int(s):int(s)+128,9] for _,f in data for s in starts(f)]).astype(np.float64)
    original=10*np.log10(np.mean(10**(channels/10)))
    return data,float(original)

def freeze_protocol():
    data,original=scenarios()
    record=dict(steps=128,trajectories=48,starts_per_file=8,seeds=BASE['seeds'],main_methods=MAIN,
        snr_db=SNRS,snr_methods=ROBUST,train=False,source_hashes=frozen_hashes(),
        original_global_mean_linear_SINR_db=original,
        snr_definition='10log10(mean(10**(exogenous_channel_dB/10))) over all scored test cycles; same dB offset applied to both whole files and all bootstrap history; no clipping',
        snr_interpretation='Existing link has SINR only; sweep changes effective noise/interference level, uses SINR as SNR proxy, not a physical signal/interference decomposition',
        snr_history='Recompute CQI, expected SE and BLER at shifted SINR, historical allocation and MCS; keep RSRP since perturbation is noise, not received signal power',
        snr_reference='Within each SNR condition: paired OLLA + original fixed PRB, same channel/start/ACK uniform draws',
        main_reference='Original recorded scenario, same initialization as previous main table; all eight methods rerun',
        no_safety='Disable quality/variance admissibility gate; retain common reward and 15ms budget fallback',
        no_calib='Disable RSRP dual temperature AND replace nine channel planning paths with mean of eight corrected prior/stress paths; safety gate and channel mean correction retained',
        identifiability='RSRP intervals are diagnostics only in current controller. Requested no-calibration variant is a compound mean-only uncertainty ablation, not a temperature-only causal ablation.',
        greedy=dict(low_db=10,high_db=15,prb_step=1,mcs='own-feedback OLLA at newly allocated per-RB SINR',test_tuning=False),
        configuration='Use previous frozen planner config and RL terminal checkpoints at all SNRs; no refit, no new training',
        safety_floor=BASE['safety_floor'],budget_ms=BASE['budget_ms'],fresh_blind_holdout=False,
        attribution='Base predictor: frezazadeh/agentic-wm_ms3m; CT-WM system wrappers are separate.',
        limits='Fixed-total-power logistic single-user link abstraction, no HARQ retransmission/queue/interference coupling; 16 starts x3 seeds are not 48 independent networks')
    write_json(OUT/'protocol.json',record)

def summarize(d):
    rows=[];episodes=[]
    for method,g in d.groupby('method',sort=False):
        worst=1e9
        for episode,t in g.groupby('episode'):
            value=100*(t.SE.sum()/t.reference_SE.sum()-1);worst=min(worst,value)
            episodes.append(dict(method=method,episode=int(episode),SE_change_percent=value,
                PRB_saving_percent=100*(1-t.PRB.sum()/t.reference_PRB.sum()),
                SE_sum=t.SE.sum(),reference_SE_sum=t.reference_SE.sum(),PRB_sum=t.PRB.sum(),reference_PRB_sum=t.reference_PRB.sum()))
        rows.append(dict(method=method,decisions=len(g),SE_change_percent=100*(g.SE.sum()/g.reference_SE.sum()-1),
            PRB_saving_percent=100*(1-g.PRB.sum()/g.reference_PRB.sum()),worst_trajectory_percent=worst,
            mean_ms=g.decision_ms.mean(),P99_ms=g.decision_ms.quantile(.99),max_ms=g.decision_ms.max(),
            fallback_count=int(g.fallback.eq(True).sum()),timeout_count=int(g.timeout.eq(True).sum()),
            over15_count=int(g.decision_ms.ge(15).sum()),mean_BLER=g.BLER.mean(),SE_sum=g.SE.sum(),reference_SE_sum=g.reference_SE.sum(),
            reference_PRB_sum=g.reference_PRB.sum(),PRB_sum=g.PRB.sum()))
    return pd.DataFrame(rows),pd.DataFrame(episodes)

def audit_condition(d,methods,condition):
    keys=['episode','step'];a=d[d.method=='olla'].set_index(keys).sort_index()
    checks=[]
    for m in methods:
        g=d[d.method==m].set_index(keys).sort_index()
        assert len(g)==6144 and g.reset_index().episode.nunique()==48
        assert np.array_equal(g[['file','start','seed']].to_numpy(),a[['file','start','seed']].to_numpy())
        np.testing.assert_allclose(g.reference_SE,a.reference_SE,rtol=0,atol=1e-9)
        np.testing.assert_array_equal(g.reference_MCS,a.reference_MCS)
        assert (g.groupby(level=0).size()==128).all()
        assert np.isfinite(g[['SE','reference_SE','decision_ms']].to_numpy()).all()
        checks.append(m)
    assert (d[d.method=='olla'].PRB==d[d.method=='olla'].reference_PRB).all()
    assert (d[d.method=='prb'].MCS==14).all()
    np.testing.assert_allclose(a.SE,a.reference_SE,rtol=0,atol=0)
    return dict(condition=condition,matched_methods=checks,matched_seeds_starts_reference=True,steps=128,trajectories=48)

def evaluate(mode):
    torch.set_num_threads(1);before=frozen_hashes();link=PythonLink();forecast=Forecast()
    config=json.loads((PREVIOUS/'planner_config.json').read_text());apps=make_apps(link,forecast,config)
    original_data,original_mean=scenarios();methods=MAIN if mode=='main' else ROBUST
    conditions=[None] if mode=='main' else SNRS
    # No concurrent training; warm every adapter, JSON route, and graph.
    for m in methods:
        node=VirtualE2Node(link);app=apps[m];app.reset(node.reset(original_data[0][1],64,42));observation(app)
        for _ in range(10):exchange(node,app)
    torch.cuda.synchronize();audits=[]
    for snr in conditions:
        name='main' if snr is None else f'snr_{snr:02d}'
        if (OUT/f'{name}_steps.csv').exists():
            print('Existing completed condition reused:',name,flush=True)
            d=pd.read_csv(OUT/f'{name}_steps.csv')
            audits.append(audit_condition(d,methods,name));continue
        delta=0. if snr is None else float(snr-original_mean)
        data=[(n,f.copy()) for n,f in original_data]
        if snr is not None:
            for _,f in data:f[:,9]+=delta
        actual=np.concatenate([f[int(s):int(s)+128,9] for _,f in data for s in starts(f)]).astype(np.float64)
        actual_mean=float(10*np.log10(np.mean(10**(actual/10))))
        if snr is not None:assert abs(actual_mean-snr)<2e-6
        write_json(OUT/f'{name}_condition.json',dict(nominal_snr_db=snr,shift_db=delta,
            achieved_exogenous_linear_mean_db=actual_mean,original_mean_db=original_mean))
        rows=[];episode=0
        for filename,f in data:
            for start in starts(f):
                for seed in BASE['seeds']:
                    for m in np.roll(methods,episode%len(methods)):
                        node=(VirtualE2Node if snr is None else ScenarioNode)(link);app=apps[m]
                        app.reset(node.reset(f,int(start),seed));torch.manual_seed(seed)
                        if hasattr(app,'seed'):app.seed(seed)
                        for step in range(128):
                            r,_,_=exchange(node,app)
                            r.update(method=m,file=filename,start=int(start),seed=seed,step=step+1,episode=episode,condition=name)
                            if 'timeout' not in r:r['timeout']=False
                            rows.append(r)
                    episode+=1
                    if episode%16==0:print(name,'matched episodes',episode,'/48',flush=True)
        d=pd.DataFrame(rows);audits.append(audit_condition(d,methods,name))
        summary,traces=summarize(d);summary['condition']=name;traces['condition']=name
        d.to_csv(OUT/f'{name}_steps.csv',index=False)
        summary.set_index('method').loc[methods].reset_index().to_csv(OUT/f'{name}_summary.csv',index=False)
        traces.to_csv(OUT/f'{name}_trajectories.csv',index=False)
        print(name,'complete\n'+summary[['method','SE_change_percent','PRB_saving_percent','worst_trajectory_percent','mean_ms','P99_ms','timeout_count']].to_string(index=False),flush=True)
    assert before==frozen_hashes(),'Frozen file modified'
    write_json(OUT/f'{mode}_audit.json',dict(conditions=audits,hashes_unchanged=True,hashes=before,training_invoked=False))

class RecordedForecast:
    def __init__(self,paths,intervals):self.paths=paths;self.intervals=intervals
    def predict(self,w,hold,config,cache=False):return self.paths.copy(),self.intervals.copy(),0.

def temperature_check():
    """Same 6144 recorded KPMs, same paths, same planner RNG; only T differs.

    Disable wall-clock timeout for this causal check (latency is not evaluated).
    Mean/stress paths remain unchanged, so risk gate inputs are identical.
    """
    torch.set_num_threads(1);torch.manual_seed(42);link=PythonLink();forecast=Forecast()
    config=json.loads((PREVIOUS/'planner_config.json').read_text());d=pd.read_csv(OUT/'main_steps.csv')
    d=d[d.method=='ctwm'];data=dict(frames('test'));count=0;differences=0;interval_differences=0
    maximum=0.;matches=[];interval_records=[]
    for episode,g in d.groupby('episode',sort=True):
        first=g.iloc[0];node=VirtualE2Node(link);bootstrap=node.reset(data[first.file],int(first.start),int(first.seed))
        context=XAppContext(link);context.reset(bootstrap)
        placeholder=np.zeros((9,8));iv=np.zeros(4)
        a=MPPIXApp(link,RecordedForecast(placeholder,iv),config,cache=True)
        b=MPPIXApp(link,RecordedForecast(placeholder,iv),config,cache=True)
        for app in [a,b]:app.reset(bootstrap);app.seed(int(first.seed))
        torch.manual_seed(int(first.seed));local=0
        for r in g.itertuples():
            k=node.last_indication;context.receive(k)
            paths,cal,_=forecast.predict(context.window(),context.hold,config)
            center=(cal[0]+cal[1])/2;h50=(cal[1]-cal[0])/2/np.sqrt(BASE['T50']);h95=(cal[3]-cal[2])/2/np.sqrt(BASE['T95'])
            raw=np.array([center-h50,center+h50,center-h95,center+h95])
            actual=float(data[first.file][int(first.start)+r.step-1,7])
            for label,iv in [('dual_temperature',cal),('unscaled',raw)]:
                interval_records.append(dict(kind=label,covered50=bool(iv[0]<=actual<=iv[1]),covered95=bool(iv[2]<=actual<=iv[3]),
                    width50=float(iv[1]-iv[0]),width95=float(iv[3]-iv[2])))
            a.forecast=RecordedForecast(paths,cal);b.forecast=RecordedForecast(paths,raw)
            ra,da=a.decide(k);rb,db=b.decide(k)
            same=(ra.prb_adjustment,ra.mcs_index)==(rb.prb_adjustment,rb.mcs_index)
            differences+=not same;local+=not same;count+=1
            interval_differences+=not np.array_equal(cal,raw);maximum=max(maximum,float(np.max(abs(cal-raw))))
            # Replay the original full controller's actual action, not either
            # counterfactual action; thus every pair sees the same recorded KPM.
            rc=context.control(int(r.PRB),int(r.MCS));_,replayed=node.execute(rc)
            assert abs(replayed['SE']-r.SE)<1e-10,'Recorded state replay mismatch'
        matches.append(dict(episode=int(episode),mismatches=local))
    assert count==6144 and differences==0
    intervals=pd.DataFrame(interval_records)
    write_json(OUT/'temperature_only_audit.json',dict(decisions=count,action_differences=differences,
        interval_differences=interval_differences,max_interval_change_db=maximum,
        rsrp_intervals_same_states=intervals.groupby('kind')[['covered50','covered95','width50','width95']].mean().reset_index().to_dict(orient='records'),
        same_KPM_paths_planner_rng=True,budget_disabled_for_causal_check=True,
        conclusion='Dual-temperature RSRP intervals do not enter current control decisions; necessity for control cannot be claimed.',episodes=matches))
    print('Temperature-only causal check:',count,'paired decisions,',differences,'action differences',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','main','snr','temperature']);args=p.parse_args()
    if args.mode=='freeze':freeze_protocol()
    elif args.mode=='temperature':temperature_check()
    else:evaluate(args.mode)

if __name__=='__main__':
    try:main()
    except Exception:
        with (OUT/'errors_full.log').open('a',encoding='utf-8') as f:traceback.print_exc(file=f)
        raise
