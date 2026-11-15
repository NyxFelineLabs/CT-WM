"""Control-dimension ablations of frozen safe CT-WM; no training."""
import json
import traceback
import numpy as np
import pandas as pd
import torch
from e2_sim import BASE,PythonLink,VirtualE2Node,frames,starts,write_json
from e2_xapps import Forecast,MPPIXApp
from e2_experiment import exchange
from e2_supplement import OUT as SUPPLEMENT,PREVIOUS,frozen_hashes,summarize

OUT=SUPPLEMENT/'control_dimensions'
OUT.mkdir(parents=True,exist_ok=True)
METHODS=['only_prb','only_mcs']

class OnlyPRB(MPPIXApp):
    def __init__(self,link,forecast,config):
        super().__init__(link,forecast,config,joint=False,safe=True)
    def control(self,p,m):
        assert m==14,'Only-PRB fallback must also keep MCS=14'
        return super().control(p,m)

class OnlyMCS(MPPIXApp):
    def __init__(self,link,forecast,config):
        super().__init__(link,forecast,config,joint=True,safe=True)
    def trajectories(self,delta,paths):
        # Projection occurs before scoring, proposal comparison and safety
        # checks. Fallback keeps hold PRB too; no post-hoc action correction.
        projected=delta.copy();projected[:,:,0]=0
        return super().trajectories(projected,paths)
    def control(self,p,m):
        assert p==self.hold,'Only-MCS search and fallback must keep initial PRB'
        return super().control(p,m)

def main():
    torch.set_num_threads(1);before=frozen_hashes()
    protocol=json.loads((SUPPLEMENT/'protocol.json').read_text())
    assert before==protocol['source_hashes']
    write_json(OUT/'protocol.json',dict(methods=METHODS,steps=128,trajectories=48,seeds=BASE['seeds'],
        only_prb='Same MPPI and safety gate; MCS fixed 14 in candidates, executed actions, and fallback',
        only_mcs='Same MPPI and safety gate; delta-PRB projected to zero before scoring; initial PRB held in execution and fallback',
        shared=dict(horizon=8,population=32,prior_samples=8,safety_floor=BASE['safety_floor'],budget_ms=BASE['budget_ms']),
        fallback='Restricted to each variant action space; the unchanged predictive gate cannot guarantee feasibility of a restricted fallback',
        baseline='Reuse full CT-WM and paired OLLA from supplement/main_steps.csv; same data/starts/seeds; separate timing run',
        hashes=before,training=False,test_tuning=False))
    link=PythonLink();forecast=Forecast();config=json.loads((PREVIOUS/'planner_config.json').read_text())
    apps=dict(only_prb=OnlyPRB(link,forecast,config),only_mcs=OnlyMCS(link,forecast,config));data=frames('test')
    for app in apps.values():
        n=VirtualE2Node(link);app.reset(n.reset(data[0][1],64,42))
        for _ in range(10):exchange(n,app)
    torch.cuda.synchronize();rows=[];episode=0
    for filename,f in data:
        for start in starts(f):
            for seed in BASE['seeds']:
                for method in np.roll(METHODS,episode%2):
                    app=apps[method];node=VirtualE2Node(link);app.reset(node.reset(f,int(start),seed));app.seed(seed);torch.manual_seed(seed)
                    for step in range(128):
                        r,_,_=exchange(node,app)
                        r.update(method=method,file=filename,start=int(start),seed=seed,episode=episode,step=step+1,condition='original')
                        rows.append(r)
                episode+=1
                if episode%16==0:print('Control dimensions: matched episodes',episode,'/48',flush=True)
    d=pd.DataFrame(rows)
    reference=pd.read_csv(SUPPLEMENT/'main_steps.csv');a=reference[reference.method=='olla'].set_index(['episode','step']).sort_index()
    for method in METHODS:
        g=d[d.method==method].set_index(['episode','step']).sort_index()
        assert len(g)==6144 and g.groupby(level=0).size().eq(128).all()
        np.testing.assert_array_equal(g[['file','start','seed']].to_numpy(),a[['file','start','seed']].to_numpy())
        np.testing.assert_allclose(g.reference_SE,a.reference_SE,rtol=0,atol=1e-9)
        np.testing.assert_array_equal(g.reference_MCS,a.reference_MCS)
        assert np.isfinite(g[['SE','decision_ms']].to_numpy()).all()
    assert d[d.method=='only_prb'].MCS.eq(14).all()
    assert d[d.method=='only_mcs'].PRB.eq(d[d.method=='only_mcs'].reference_PRB).all()
    assert d[d.method=='only_mcs'].abs_delta_PRB.eq(0).all()
    assert before==frozen_hashes()
    summary,traces=summarize(d)
    d.to_csv(OUT/'steps.csv',index=False);traces.to_csv(OUT/'trajectories.csv',index=False)
    summary.set_index('method').loc[METHODS].reset_index().to_csv(OUT/'summary.csv',index=False)
    full=pd.read_csv(SUPPLEMENT/'main_summary.csv');combined=pd.concat([full[full.method=='ctwm'],summary],ignore_index=True)
    combined.to_csv(OUT/'control_dimension_summary.csv',index=False)
    # Paired descriptive intervals use 16 file/start blocks; all three ACK
    # seeds in a block are aggregated before resampling.
    rng=np.random.default_rng(20261112);idx=rng.integers(0,16,(4000,16));keys=['file','start'];cols=['SE','reference_SE','PRB','reference_PRB']
    base=reference[reference.method=='ctwm'].groupby(keys)[cols].sum().sort_index();bs=base.to_numpy()[idx].sum(1);stats=[]
    for method in METHODS:
        c=d[d.method==method].groupby(keys)[cols].sum().sort_index().loc[base.index].to_numpy()[idx].sum(1)
        stats.append(dict(method=method,SE_full_minus_variant_95pp=np.quantile(100*(bs[:,0]-c[:,0])/bs[:,1],[.025,.975]).tolist(),
            PRB_saving_full_minus_variant_95pp=np.quantile(100*(c[:,2]-bs[:,2])/bs[:,3],[.025,.975]).tolist()))
    write_json(OUT/'paired_bootstrap.json',dict(unit='16 file/start blocks, three seeds aggregated per block',records=stats,scope='Conditional descriptive comparison on existing traces'))
    write_json(OUT/'audit.json',dict(matched_seeds_starts_reference=True,decisions_per_method=6144,
        only_prb_fixed_mcs_all_decisions=True,only_mcs_fixed_prb_all_decisions=True,
        restrictions_apply_to_candidates_and_fallback=True,hashes_unchanged=True,hashes=before,no_training=True))
    print(combined.to_string(index=False),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        with (OUT/'errors_full.log').open('a',encoding='utf-8') as f:traceback.print_exc(file=f)
        raise
