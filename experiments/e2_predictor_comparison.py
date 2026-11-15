"""Round2 matched forecasting comparison; frozen model/planner, no training.

Both controllers retain the existing joint MPPI, gate, fallback and budget.
Only the forecasting adapter and its validation-derived error scale differ.
"""
import json,time,traceback,platform,sys,os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from e2_sim import ROOT,BASE,PythonLink,VirtualE2Node,frames,starts,write_json,sha
from e2_xapps import Forecast,MPPIXApp
from e2_experiment import exchange
from e2_supplement import frozen_hashes,summarize

OUT=ROOT/'runs/predictor_comparison'
OUT.mkdir(parents=True,exist_ok=True)
PREVIOUS=ROOT/'runs/e2_python'
STRESS=np.array([-1.645,-1.,-.5,-.2,.2,.5,1.,1.645])[:,None]

class PersistenceForecast:
    """Hold the measured exogenous channel; validation horizon bias and noise.

    RSRP interval diagnostics are unavailable (zero placeholders satisfy the
    unchanged planner's logging interface); they never enter action selection.
    """
    def predict(self,w,hold,config,cache=False):
        begin=time.perf_counter()
        current=float(w[-1,9]-10*np.log10(hold/w[-1,5]))
        center=current+np.array(config['bias'])
        paths=np.broadcast_to(center,(9,8)).copy()
        paths[1:]+=STRESS*np.array(config['residual_sigma'])[None]
        return paths,np.zeros(4),1000*(time.perf_counter()-begin)

def prepare():
    """Use the same validation origins/fit split; do not optimize on test."""
    cfg=json.loads((PREVIOUS/'planner_config.json').read_text())
    name,f=frames('val')[0]
    origins=np.linspace(64,len(f)-136,192,dtype=int)
    current=f[origins-1,9,None]
    targets=np.array([f[i:i+8,9] for i in origins])
    residual=targets-current
    bias=residual[:128].mean(0)
    centered=residual-bias
    sigma=centered[:128].std(0,ddof=1)
    persistence=dict(cfg,alpha=0.,bias=bias.tolist(),residual_sigma=sigma.tolist(),
        forecast_mse=float(np.mean(centered[128:]**2)))
    archived=json.loads((PREVIOUS/'forecast_validation.json').read_text())
    p0=next(x for x in archived['candidates'] if x['alpha']==0.)
    np.testing.assert_allclose(bias,p0['bias'],rtol=0,atol=2e-6)
    np.testing.assert_allclose(sigma,p0['residual_sigma'],rtol=0,atol=2e-6)
    write_json(OUT/'persistence_config.json',persistence)
    write_json(OUT/'protocol.json',dict(methods=['learned','persistence'],steps=128,trajectories=48,
        seeds=BASE['seeds'],horizon=8,population=32,execute_first_only=True,training=False,
        frozen_config=cfg,persistence_config=persistence,validation_file=name,
        validation_origins=origins.tolist(),fitting_origins=128,validation_check_origins=64,
        simple_forecast='current exogenous channel + horizon bias; eight validation-residual stress paths; no learned predictor',
        matched='Same MPPIXApp implementation, reward, variance threshold, safety floor, joint action space, fallback, nominal initialization, planner RNG, channel starts and feedback draws',
        error_scale='Each variant uses its own validation-only horizon residual scale. Persistence uses identical fitting origins and no grid selection; shared gate threshold/reward are kept from frozen learned-controller configuration.',
        limitations='Gate threshold/reward were originally selected with learned forecasts, not symmetrically retuned. Test files previously exposed. No new independent holdout or algorithm-wide conclusion.',
        rsrp='Persistence has no RSRP interval prediction; zero logging placeholders are not evaluated and do not enter the unchanged gate.',
        timing='Same exchange KPM JSON -> RC JSON boundary; no forecast output caching; 15ms operator-boundary checks kept. Historical main table timing is a different run.',
        source_hashes=frozen_hashes(),new_script_sha256=sha(Path(__file__))))
    return cfg,persistence

def main():
    assert not (OUT/'steps.csv').exists(),'Do not overwrite a completed comparison'
    torch.set_num_threads(1)
    before=frozen_hashes();cfg,pcfg=prepare()
    link=PythonLink();forecast=Forecast()
    apps=dict(learned=MPPIXApp(link,forecast,cfg),persistence=MPPIXApp(link,PersistenceForecast(),pcfg))
    data=frames('test')
    for app in apps.values():
        node=VirtualE2Node(link);app.reset(node.reset(data[0][1],64,42));app.seed(42);torch.manual_seed(42)
        for _ in range(10):exchange(node,app)
    torch.cuda.synchronize();rows=[];episode=0;begin=time.perf_counter()
    for filename,f in data:
        for start in starts(f):
            for seed in BASE['seeds']:
                for name in np.roll(['learned','persistence'],episode%2):
                    node=VirtualE2Node(link);app=apps[name]
                    app.reset(node.reset(f,int(start),seed));app.seed(seed);torch.manual_seed(seed)
                    for step in range(128):
                        r,_,_=exchange(node,app)
                        r.update(method=name,file=filename,start=int(start),seed=seed,episode=episode,step=step+1,
                            condition='original',rsrp_intervals_available=name=='learned')
                        rows.append(r)
                episode+=1
                if episode%16==0:print(f'Matched predictor episodes {episode}/48',flush=True)
    d=pd.DataFrame(rows);a=d[d.method=='learned'].set_index(['episode','step']).sort_index()
    b=d[d.method=='persistence'].set_index(['episode','step']).sort_index()
    assert len(a)==len(b)==6144
    np.testing.assert_array_equal(a[['file','start','seed']],b[['file','start','seed']])
    np.testing.assert_allclose(a.reference_SE,b.reference_SE,rtol=0,atol=1e-10)
    np.testing.assert_array_equal(a.reference_MCS,b.reference_MCS)
    assert np.isfinite(d[['SE','reference_SE','decision_ms']]).all().all()
    summary,traces=summarize(d)
    summary['fallback_percent']=100*summary.fallback_count/6144
    d.to_csv(OUT/'steps.csv',index=False);traces.to_csv(OUT/'trajectories.csv',index=False)
    summary.to_csv(OUT/'summary.csv',index=False)
    per_file=[]
    for (method,file),g in d.groupby(['method','file']):
        per_file.append(dict(method=method,file=file,SE_change_percent=100*(g.SE.sum()/g.reference_SE.sum()-1),
            PRB_saving_percent=100*(1-g.PRB.sum()/g.reference_PRB.sum())))
    pd.DataFrame(per_file).to_csv(OUT/'per_file.csv',index=False)
    keys=['file','start'];cols=['SE','reference_SE','PRB','reference_PRB']
    aa=d[d.method=='learned'].groupby(keys)[cols].sum().sort_index()
    bb=d[d.method=='persistence'].groupby(keys)[cols].sum().sort_index().loc[aa.index]
    idx=np.random.default_rng(20261112).integers(0,16,(4000,16))
    ra=aa.to_numpy()[idx].sum(1);rb=bb.to_numpy()[idx].sum(1)
    write_json(OUT/'paired_bootstrap.json',dict(
        SE_learned_minus_persistence_95pp=np.quantile(100*(ra[:,0]-rb[:,0])/ra[:,1],[.025,.975]).tolist(),
        PRB_saving_learned_minus_persistence_95pp=np.quantile(100*(rb[:,2]-ra[:,2])/ra[:,3],[.025,.975]).tolist(),
        scope='Conditional descriptive intervals, 16 file/start blocks with three feedback seeds aggregated; does not remove within-file dependence or prior test exposure'))
    cpu=platform.processor() or platform.machine()
    write_json(OUT/'runtime.json',dict(cpu=cpu,gpu=torch.cuda.get_device_name(0),python=sys.version,platform=platform.platform(),
        torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,cuda=torch.version.cuda,
        torch_intraop_threads=torch.get_num_threads(),elapsed_seconds=time.perf_counter()-begin))
    assert before==frozen_hashes()
    write_json(OUT/'audit.json',dict(status='PASS',matched_starts_seeds_reference=True,trajectories=48,decisions_per_method=6144,
        validation_only_error_scale=True,training=False,frozen_sources_unchanged=True,source_hashes=before))
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        with (OUT/'errors_full.log').open('a',encoding='utf-8') as stream:traceback.print_exc(file=stream)
        raise
