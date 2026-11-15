"""Python-only matched KPM/RC experiment. No native link dependencies."""
import argparse
import time
import json
import numpy as np
import pandas as pd
import torch
from e2_sim import (OUT,BASE,PythonLink,VirtualE2Node,XAppContext,
    E2SMKPMIndication as KPM,E2SMRCControl as RC,score,freeze,guard,frames,starts,write_json,sha,ROOT)
from e2_xapps import Forecast,OLLAXApp,MPPIXApp,RLXApp,observation
from simple_olla import SimpleOLLA

METHODS=['olla','prb','sac','ppo','ctwm']

def exchange(node,app):
    begin=time.perf_counter_ns()
    indication=KPM.read(node.last_indication.wire())
    action,diag=app.decide(indication)
    action=RC.read(action.wire())
    elapsed=(time.perf_counter_ns()-begin)/1e6
    wire,metrics=node.execute(action)
    return dict(metrics,decision_ms=elapsed,**diag),wire,action

def verify():
    link=PythonLink();o=SimpleOLLA()
    initial=o.select_mcs(17,link.threshold10)
    o.update(False);assert o.select_mcs(17,link.threshold10)<=initial
    o=SimpleOLLA()
    for a in [True]*9+[False]:o.update(a)
    assert abs(o.offset_db)<1e-12
    for sinr in np.linspace(0,25,51):
        assert SimpleOLLA().select_mcs(sinr,link.threshold10)==link.select(sinr,20,20,0)
    _,f=frames('val')[0];node=VirtualE2Node(link);app=OLLAXApp(link)
    app.reset(node.reset(f,64,42));rows=[]
    for _ in range(128):
        row,_,action=exchange(node,app);rows.append(row)
        assert row['PRB']==node.hold and row['SE']==row['reference_SE']
    try:node.execute(action)
    except ValueError:pass
    else:raise AssertionError('Replay accepted')
    assert sha(ROOT/'src/wm_ms3m/core.py')=='05a723ee538cca0374e9823ac08a285406f05b9bde9e8851908c0ade5372d6c1'
    manifest=json.loads((ROOT/'configs/weights_manifest.json').read_text())
    expected=manifest['files']['artifacts/file_split_width_93/best.pt']['sha256']
    assert sha(ROOT/'artifacts/file_split_width_93/best.pt')==expected
    write_json(OUT/'verification.json',dict(olla_negative_feedback=True,nine_ACK_one_NACK_net_zero=True,
        reference_exact_128_steps=True,rc_replay_rejected=True,core_and_checkpoint_unchanged=True))
    freeze();print('Python loop and OLLA verification passed',flush=True)

def calibrate():
    torch.manual_seed(42);link=PythonLink();forecast=Forecast();_,f=frames('val')[0]
    raw=[];target=[];current=[]
    for i in np.linspace(64,len(f)-136,192,dtype=int):
        node=VirtualE2Node(link);node.reset(f,int(i),42)
        w=np.array([KPM.read(s).kpis for s in node.bootstrap],np.float32)
        paths,_,_=forecast.raw(w,node.hold)
        raw.append(paths[0]);target.append(f[i:i+8,9]);current.append(f[i-1,9])
    raw=np.array(raw);target=np.array(target);current=np.array(current)[:,None]
    trials=[]
    # Disjoint validation blocks: first 128 fit, remaining 64 select blend.
    for alpha in [0.,.25,.5,.75,1.]:
        bias=np.mean(target[:128]-(current+alpha*(raw-current))[:128],axis=0)
        residual=target-(current+alpha*(raw-current)+bias)
        trials.append(dict(alpha=alpha,bias=bias.tolist(),residual_sigma=np.std(residual[:128],axis=0,ddof=1).tolist(),
            forecast_mse=float(np.mean(residual[128:]**2))))
    best=min(trials,key=lambda v:v['forecast_mse'])
    # Stress variance cutoff derived from validation radio responses, not test.
    stresses=np.array([-1.645,-1.,-.5,-.2,.2,.5,1.,1.645])[:,None]
    variations=[]
    for i in np.linspace(64,len(f)-136,128,dtype=int):
        p=int(np.clip(round(float(f[i-1,5])),13,25));ch=float(f[i-1,9]);m=link.select(ch,p,p,0)
        h=ch+stresses*np.array(best['residual_sigma'])[None]
        se,_=link.se(h,p,m,p)
        variations.append(float((se.var(0)/np.maximum(abs(se.mean(0)),.05)**2).max()))
    best['variance_limit']=float(np.quantile(variations,.99));best['penalty']=1.;best['resource']=1.
    write_json(OUT/'forecast_validation.json',dict(candidates=trials,selected=best,validation_file=frames('val')[0][0]))
    # Choose from a predeclared small reward grid on validation trajectories.
    configs=[];records=[]
    for penalty,resource in [(1.,1.),(1.,1.5),(2.,1.),(2.,1.5)]:
        config=dict(best,penalty=penalty,resource=resource)
        results=[]
        for start in starts(f):
            node=VirtualE2Node(link);app=MPPIXApp(link,forecast,config,cache=True)
            app.reset(node.reset(f,int(start),42));app.seed(42);torch.manual_seed(42)
            for step in range(128):
                r,_,_=exchange(node,app);results.append(dict(r,start=int(start)))
        d=pd.DataFrame(results);se=100*(d.SE.sum()/d.reference_SE.sum()-1);saving=100*(1-d.PRB.sum()/d.reference_PRB.sum())
        worst=min(100*(g.SE.sum()/g.reference_SE.sum()-1) for _,g in d.groupby('start'))
        violation=max(0,-2-se)+max(0,se-2)+max(0,-5-worst)
        records.append(dict(penalty=penalty,resource=resource,SE_percent=se,PRB_percent=saving,worst_percent=worst,violation=violation))
        configs.append(config);print('validation',records[-1],flush=True)
    idx=min(range(len(records)),key=lambda i:(records[i]['violation']>0,records[i]['violation'],-records[i]['PRB_percent']))
    write_json(OUT/'planner_config.json',configs[idx]);write_json(OUT/'reward_validation.json',dict(grid=records,selected=idx,
        selection='Minimize validation quality violation, then maximize PRB saving; no test tuning',timing='Validation cache only; disabled in final evaluation'))
    print('Validation configuration frozen',flush=True)

def train(algorithm):
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import SAC,PPO
    config=json.loads((OUT/'planner_config.json').read_text());torch.set_num_threads(1)
    class Radio(gym.Env):
        def __init__(self):
            super().__init__();self.data=frames('train');self.link=PythonLink();self.node=VirtualE2Node(self.link);self.app=XAppContext(self.link);self.count=0
            self.action_space=spaces.Box(-1.,1.,(2,),np.float32);self.observation_space=spaces.Box(-np.inf,np.inf,(419,),np.float32)
        def reset(self,seed=None,options=None):
            super().reset(seed=seed);_,f=self.data[int(self.np_random.integers(len(self.data)))];s=int(self.np_random.integers(64,len(f)-128+1))
            self.app.reset(self.node.reset(f,s,int(self.np_random.integers(2**31))));return observation(self.app),{}
        def step(self,a):
            prev=self.app.current;delta=np.rint(2*np.asarray(a)).astype(int)
            p=int(np.clip(prev.current_prb+delta[0],13,25));m=int(np.clip(prev.current_mcs+delta[1],0,28))
            rc=RC.read(self.app.control(p,m).wire());wire,r=self.node.execute(rc);self.app.receive(KPM.read(wire));self.count+=1
            reward=float(score(r['SE'],r['reference_SE'],p,self.app.hold,p-prev.current_prb,m-prev.current_mcs,config))
            return observation(self.app),reward,False,self.node.t>=128,{}
    env=Radio();begin=time.perf_counter();common=dict(env=env,policy='MlpPolicy',policy_kwargs=dict(net_arch=[128,128]),seed=42,verbose=0,device='cpu')
    model=SAC(**common,buffer_size=50000,learning_starts=2000,batch_size=256,learning_rate=3e-4,train_freq=4,gradient_steps=1,gamma=.99,tau=.005) if algorithm=='sac' else PPO(**common,n_steps=1000,batch_size=250,n_epochs=10,learning_rate=3e-4,gamma=.99,gae_lambda=.95,clip_range=.2)
    model.learn(total_timesteps=50000);assert env.count==50000
    folder=OUT/'rl';folder.mkdir(parents=True,exist_ok=True);model.save(folder/algorithm)
    write_json(folder/f'{algorithm}_training.json',dict(interactions=env.count,seconds=time.perf_counter()-begin,seed=42,
        network=[128,128],training_files=[n for n,_ in frames('train')],terminal_checkpoint=True,
        policy_action='Normalized deltas round(2*a); RC boundary [delta_PRB,absolute_MCS]',reward=config,
        torch=torch.__version__,sb3=__import__('stable_baselines3').__version__))
    print(algorithm,'50k training complete',time.perf_counter()-begin,flush=True)

def evaluate():
    from stable_baselines3 import SAC,PPO
    torch.set_num_threads(1);link=PythonLink();forecast=Forecast();config=json.loads((OUT/'planner_config.json').read_text())
    apps=dict(olla=OLLAXApp(link),prb=MPPIXApp(link,forecast,config,joint=False,safe=False),
        sac=RLXApp(link,SAC.load(OUT/'rl/sac',device='cpu')),ppo=RLXApp(link,PPO.load(OUT/'rl/ppo',device='cpu')),
        ctwm=MPPIXApp(link,forecast,config))
    data=frames('test');all_rows=[];examples=[]
    # Warm all adapters and graph before timing; no RL training runs concurrently.
    for app in apps.values():
        n=VirtualE2Node(link);app.reset(n.reset(data[0][1],64,42));observation(app)
        for _ in range(10):exchange(n,app)
    torch.cuda.synchronize()
    # Alternate method order across episodes, preserving every state/seed pair.
    episode=0
    for filename,f in data:
        for start in starts(f):
            for seed in BASE['seeds']:
                for method in np.roll(METHODS,episode%len(METHODS)):
                    node=VirtualE2Node(link);app=apps[method];app.reset(node.reset(f,int(start),seed));torch.manual_seed(seed)
                    if hasattr(app,'seed'):app.seed(seed)
                    rows=[]
                    for step in range(128):
                        r,wire,rc=exchange(node,app);r.update(method=method,file=filename,start=int(start),seed=seed,step=step+1,episode=episode)
                        rows.append(r)
                        if episode==0 and step==0:examples.append(dict(method=method,kpm=json.loads(wire),rc=json.loads(rc.wire())))
                    all_rows+=rows
                episode+=1
                if episode%8==0:
                    pd.DataFrame(all_rows).to_csv(OUT/'steps_partial.csv',index=False)
                    print('Completed matched episodes',episode,'/48',flush=True)
    d=pd.DataFrame(all_rows);d.to_csv(OUT/'steps.csv',index=False);write_json(OUT/'interface_examples.json',examples)
    summaries=[];traces=[]
    for method,g in d.groupby('method',sort=False):
        worst=1e9
        for ep,t in g.groupby('episode'):
            change=100*(t.SE.sum()/t.reference_SE.sum()-1);worst=min(worst,change)
            traces.append(dict(method=method,episode=int(ep),SE_change_percent=change,PRB_saving_percent=100*(1-t.PRB.sum()/t.reference_PRB.sum()),
                mean_ms=t.decision_ms.mean(),P99_ms=t.decision_ms.quantile(.99),max_ms=t.decision_ms.max()))
        summaries.append(dict(method=method,decisions=len(g),SE_change_percent=100*(g.SE.sum()/g.reference_SE.sum()-1),
            PRB_saving_percent=100*(1-g.PRB.sum()/g.reference_PRB.sum()),worst_trajectory_percent=worst,
            mean_ms=g.decision_ms.mean(),P99_ms=g.decision_ms.quantile(.99),max_ms=g.decision_ms.max(),
            fallback_count=int(g.fallback.fillna(False).sum()),timeout_count=int(g.get('timeout',pd.Series(False,index=g.index)).fillna(False).sum()),
            over15_count=int((g.decision_ms>=15).sum()),mean_abs_PRB_change=g.abs_delta_PRB.mean(),mean_abs_MCS_change=g.abs_delta_MCS.mean()))
    pd.DataFrame(summaries).set_index('method').loc[METHODS].reset_index().to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(traces).to_csv(OUT/'trajectories.csv',index=False)
    freeze();print(pd.read_csv(OUT/'summary.csv').to_string(index=False),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['verify','calibrate','sac','ppo','evaluate']);a=p.parse_args().stage
    {'verify':verify,'calibrate':calibrate,'sac':lambda:train('sac'),'ppo':lambda:train('ppo'),'evaluate':evaluate}[a]()

if __name__=='__main__':guard(main)
