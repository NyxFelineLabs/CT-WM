"""xApps operate only on KPMs and emit RC objects; shared-channel fast MPPI."""
import json
import time
import numpy as np
import torch
from e2_sim import OUT,BASE,PythonLink,XAppContext,E2SMKPMIndication,score
from ct_common import load_bundle
from planner_runtime import make_executor

class Forecast:
    """Eight channel paths share one frozen NN forecast; controls affect PHY.

    Exogenous replay channel is independent of controls, so no candidate-specific
    16/32-population neural rollout is needed. Eight prior draws plus prior mean.
    """
    def __init__(self):
        self.b=load_bundle('L32');self.ex=make_executor(self.b)
        self.input=torch.ones(1,32,13,device='cuda');self.eps=torch.zeros(8,9,self.b.cfg.z_dim,device='cuda')
        self.stream=torch.cuda.Stream();self.stream.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(),torch.cuda.stream(self.stream):
            for _ in range(3):self._forward()
        self.stream.synchronize();self.graph=torch.cuda.CUDAGraph()
        with torch.no_grad(),torch.cuda.graph(self.graph,stream=self.stream):self.result=self._forward()
        self.stream.synchronize();self.cache={};self.elapsed={}
    def _forward(self):
        w=self.input.expand(9,-1,-1);model=self.ex.model;channels=[];rsrp=None
        for t in range(8):
            x=(w-self.ex.x_mean)/self.ex.x_std;d=model.forward_d(x,x[:,:,5:6]);mu,lv=model._split_gauss(model.prior_net(d))
            z=mu+(.5*lv).exp()*self.eps[t];hz=torch.cat([d,z],-1)
            y=model.full_decoder(hz)*self.ex.y_full_std+self.ex.y_full_mean;channels.append(y[:,9])
            if t==0:
                a,v=model.target_head(hz).chunk(2,-1);a=a+model.skip_gain.tanh()*model.ar_head(x[:,-1])
                center=a[1:].mean()*float(self.b.meta['y_std'][0])+float(self.b.meta['y_mean'][0])
                std=(v[1:].exp().mean()+a[1:].var(unbiased=False)).sqrt()*float(self.b.meta['y_std'][0])
                h50=.6744897502*(BASE['T50']**.5)*std;h95=1.959963985*(BASE['T95']**.5)*std
                rsrp=torch.stack([center-h50,center+h50,center-h95,center+h95])
            # Exogenous channel projection: hold current PRB history, not future
            # user controls, while keeping all frozen model weights unchanged.
            y=torch.cat([y[:,:5],w[:,-1,5:6],y[:,6:]],-1)
            w=torch.cat([w[:,1:],y[:,None]],1)
        return torch.stack(channels,-1),rsrp
    @torch.no_grad()
    def raw(self,w,hold,cache=False):
        begin=time.perf_counter();physical=w.copy();physical[:,9]-=10*np.log10(hold/physical[:,5])
        key=physical.tobytes()
        if cache and key in self.cache:return self.cache[key]
        self.input.copy_(torch.as_tensor(physical[None],device='cuda'))
        self.eps.normal_();self.eps[:,0].zero_();self.graph.replay()
        paths=self.result[0].cpu().numpy().copy();intervals=self.result[1].cpu().numpy().copy()
        result=(paths,intervals,1000*(time.perf_counter()-begin))
        if cache:self.cache[key]=result
        return result
    def predict(self,w,hold,config,cache=False):
        paths,intervals,cost=self.raw(w,hold,cache);current=float(w[-1,9]-10*np.log10(hold/w[-1,5]))
        alpha=config['alpha'];bias=np.array(config['bias']);sigma=np.array(config['residual_sigma'])
        forecast=current+alpha*(paths-current)+bias[None]
        # Do not invent epistemic coverage: eight prior residual paths plus
        # validation prediction error scale, a planning stress set only.
        stresses=np.array([-1.645,-1.,-.5,-.2,.2,.5,1.,1.645])[:,None]
        forecast[1:]+=stresses*sigma[None]
        return forecast,intervals,cost

class OLLAXApp(XAppContext):
    def decide(self,kpm):
        self.receive(kpm)
        m=self.offset.select_mcs(self.channel(),self.link.threshold10)
        return self.control(self.hold,m),dict(reason='olla',fallback=False)

class MPPIXApp(XAppContext):
    def __init__(self,link,forecast,config,joint=True,safe=True,cache=False):
        super().__init__(link);self.forecast=forecast;self.config=config;self.joint=joint;self.safe=safe;self.cache=cache
    def reset(self,bootstrap):
        super().reset(bootstrap);self.nominal=np.zeros((8,2));self.rng=np.random.default_rng(42)
    def seed(self,seed):self.rng=np.random.default_rng(seed)
    def trajectories(self,delta,paths):
        p=np.clip(self.current.current_prb+delta[:,:,0].cumsum(1),13,25).astype(int)
        m=np.clip(self.current.current_mcs+delta[:,:,1].cumsum(1),0,28).astype(int) if self.joint else np.full(p.shape,self.fixed_mcs)
        se,bl=self.link.se(paths[:,None,:],p[None],m[None],self.hold)
        rp=self.hold;previous=np.concatenate([np.full((9,1),self.channel()),paths[:,:-1]],1)
        rm=self.link.select(previous,np.full(previous.shape,rp),rp,self.offset.offset_db)
        ref,_=self.link.se(paths,rp,rm,rp)
        pp=np.concatenate([np.full((len(p),1),self.current.current_prb),p[:,:-1]],1)
        mm=np.concatenate([np.full((len(p),1),self.current.current_mcs),m[:,:-1]],1)
        scores=score(se,ref[:,None],p[None],rp,p[None]-pp[None],m[None]-mm[None],self.config)
        # Optimize the mean forecast; prior stresses are execution risk checks.
        return scores[0].sum(-1),p,m,se,ref
    def decide(self,kpm):
        begin=time.perf_counter();self.receive(kpm)
        fallback=self.fallback() if self.joint else (self.hold,self.fixed_mcs)
        paths,intervals,_=self.forecast.predict(self.window(),self.hold,self.config,self.cache)
        diag=dict(fallback=False,reason='accepted',timeout=False,forecast_used=True,
            rsrp50_low=float(intervals[0]),rsrp50_high=float(intervals[1]),rsrp95_low=float(intervals[2]),rsrp95_high=float(intervals[3]))
        def fail(reason):
            diag.update(fallback=True,reason=reason,timeout=reason=='budget');return self.control(*fallback),diag
        if (time.perf_counter()-begin)*1000>=BASE['budget_ms'] and not self.cache:return fail('budget')
        samples=np.rint(self.nominal[None]+self.rng.normal(0,1.5,(32,8,2))).clip(-2,2)
        samples[:5]=np.array([[0,0],[-1,1],[-2,1],[-1,2],[0,1]])[:,None]
        if not self.joint:samples[:,:,1]=0
        scores,p,m,se,ref=self.trajectories(samples,paths)
        weight=np.exp(np.clip((scores-scores.max())/.15,-60,0));weight/=weight.sum()
        proposal=np.rint((weight[:,None,None]*samples).sum(0)).clip(-2,2)
        candidates=np.concatenate([samples,proposal[None]],0)
        scores,p,m,se,ref=self.trajectories(candidates,paths)
        # All first-action hold checks are vectorized, including eight stress
        # paths and eight lookahead steps; no per-candidate Python loop.
        held,_=self.link.se(paths[1:,None,:],p[None,:,0,None],m[None,:,0,None],self.hold)
        relative=(held-BASE['safety_floor']*ref[1:,None,:])/np.maximum(np.abs(ref[1:,None,:]),.05)
        margins=np.quantile(relative,.05,axis=0).min(-1)
        variances=(held.var(0)/np.maximum(np.abs(ref[1:].mean(0)),.05)**2).max(-1)
        admissible=(margins>=0)&(variances<=self.config['variance_limit']) if self.safe else np.ones(len(scores),dtype=bool)
        chosen=int(np.argmax(np.where(admissible,scores,-np.inf))) if admissible.any() else None
        if chosen is None:return fail('quality_or_variance')
        diag.update(q05_margin=float(margins[chosen]),relative_variance=float(variances[chosen]))
        if (time.perf_counter()-begin)*1000>=BASE['budget_ms'] and not self.cache:return fail('budget')
        self.nominal=np.concatenate([candidates[chosen,1:],candidates[chosen,-1:]])
        return self.control(int(p[chosen,0]),int(m[chosen,0])),diag

_stats=None
def observation(app):
    global _stats
    if _stats is None:
        b=load_bundle('L32');_stats=np.asarray(b.meta['x_mean']),np.asarray(b.meta['x_std'])
    return np.concatenate([((app.window()-_stats[0])/_stats[1]).reshape(-1),
        [app.offset.offset_db/20,(app.hold-19)/6,app.fixed_mcs/28]]).astype(np.float32)

class RLXApp(XAppContext):
    def __init__(self,link,model):super().__init__(link);self.model=model
    def decide(self,kpm):
        self.receive(kpm);a,_=self.model.predict(observation(self),deterministic=True);d=np.rint(2*a).astype(int)
        p=int(np.clip(self.current.current_prb+d[0],13,25));m=int(np.clip(self.current.current_mcs+d[1],0,28))
        return self.control(p,m),dict(reason='rl',fallback=False)
