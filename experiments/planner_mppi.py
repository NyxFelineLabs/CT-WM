"""Reduced-population MPPI with explicit feasibility checks and graph replay."""
import json
import time
import numpy as np
import torch
from planner_runtime import OPT,make_executor
from system_common import load_bundle

H=8

class GraphMPPI:
    def __init__(self,bundle,population=32,sigma=4.,temperature=.5,enforce_se=True):
        self.b=bundle
        self.executor=make_executor(bundle)
        self.population=population
        self.sigma=sigma
        self.temperature=temperature
        self.enforce_se=enforce_se
        self.low=bundle.meta['action_bounds']['low']
        self.high=bundle.meta['action_bounds']['high']
        self.window=torch.zeros(1,32,13,device='cuda')
        self.noise=torch.zeros(population,H,device='cuda')
        self.nominal=torch.zeros(H,device='cuda')
        # initial action, remaining, completed SE, completed PRB, fixed reference SE,
        # initial flag. Budget/quality are episode totals, not drifting references.
        self.state=torch.zeros(6,device='cuda')
        self.reward_weight=bundle.reward_fn.weight.detach()
        self.indices=torch.tensor([9,10,11,12,7],device='cuda')
        self.graph=torch.cuda.CUDAGraph()
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        self.window.fill_(1.)
        self.nominal.fill_(20.)
        self.state.copy_(torch.tensor([20,8,0,0,1,1.],device='cuda'))
        with torch.no_grad(),torch.cuda.stream(stream):
            for _ in range(3): self._solve()
        torch.cuda.current_stream().wait_stream(stream)
        with torch.no_grad(),torch.cuda.graph(self.graph):
            self.result=self._solve()

    def _rollout(self,actions):
        B=actions.shape[0]
        window=self.window.expand(B,-1,-1)
        total_se=torch.zeros(B,device='cuda')
        total_prb=torch.zeros_like(total_se)
        reward=torch.zeros_like(total_se)
        minimum_se=torch.full_like(total_se,1e9)
        prev=window[:,-1,5]
        for t in range(H):
            frame=self.executor(window)
            active=(t<self.state[1]).float()
            action=actions[:,t]
            ys=(frame-self.executor.y_full_mean)/self.executor.y_full_std
            act=(action-self.executor.x_mean[5])/self.executor.x_std[5]
            features=torch.stack([ys[:,9],ys[:,10],ys[:,11],ys[:,12],act,ys[:,7]],-1)
            r=(features*self.reward_weight).sum(-1)-self.b.cfg.action_smooth_penalty*abs((action-prev)/self.executor.x_std[5])
            reward=reward+active*r
            total_se=total_se+active*frame[:,10]
            total_prb=total_prb+active*action
            minimum_se=torch.minimum(minimum_se,torch.where(active>0,frame[:,10],torch.full_like(action,1e9)))
            # Action enters history after prediction, matching frozen environment.
            next_frame=torch.cat([frame[:,:5],action[:,None],frame[:,6:]],-1)
            window=torch.cat([window[:,1:],next_frame[:,None]],1)
            prev=action
        return total_se,total_prb,reward,minimum_se

    def _feasible(self,se,prb,minse,reference):
        se_violation=torch.relu(.99*reference-(se+self.state[2]))/reference.abs().clamp_min(1e-6)
        prb_violation=torch.relu((prb+self.state[3])-.849*self.state[0]*H)/(self.state[0]*H).clamp_min(1e-6)
        if self.enforce_se:
            safe=(se_violation<=1e-6)&(minse>=-1e-6)
        else:
            safe=torch.ones_like(se,dtype=torch.bool)
        joint=safe&(prb_violation<=1e-6)
        return safe,joint,se_violation,prb_violation

    def _solve(self):
        random=self.nominal[None]+self.sigma*self.noise
        init=self.state[0]
        const=lambda v: torch.ones(H,device='cuda')*v
        ramp=torch.linspace(0,1,H,device='cuda')
        positions=torch.arange(H,device='cuda')
        tails=[torch.where(positions<self.state[1]-k,init,const(self.low)) for k in [1,2,3,4]]
        anchors=torch.stack([const(init),self.nominal,const(.84*init),const(.70*init),
                             self.low+(self.high-self.low)*ramp,
                             self.high-(self.high-self.low)*ramp]+tails)
        candidates=torch.cat([anchors,random[10:]],0).clamp(self.low,self.high)
        # Spend random samples on the remaining PRB-budget manifold, while
        # retaining unprojected anchors as safe fallbacks if the joint goal fails.
        active=(positions<self.state[1]).float()[None]
        available=.849*init*H-self.state[3]
        stochastic=candidates[10:]
        for _ in range(4):
            excess=torch.relu((stochastic*active).sum(1,keepdim=True)-available)
            stochastic=(stochastic-excess/self.state[1].clamp_min(1.)*active).clamp(self.low,self.high)
        half=(self.population-10)//2
        candidates=torch.cat([candidates[:10],stochastic[:half],candidates[10+half:]],0)
        se,prb,reward,minse=self._rollout(candidates)
        reference=self._reference(se)
        safe,joint,sev,pv=self._feasible(se,prb,minse,reference)
        eligible=torch.where(joint.any(),joint,safe)
        # Hard SE penalty + rejection; resource target is enforced when feasible.
        penalty=(1e6*sev if self.enforce_se else 0.)+1e4*pv
        scores=reward-penalty
        scores=torch.where(eligible,scores,torch.full_like(scores,-1e12))
        weights=torch.softmax((scores-scores.max())/self.temperature,0)
        proposal=(weights[:,None]*candidates).sum(0)
        ps,pp,pr,pm=self._rollout(proposal[None])
        psafe,pjoint,_,_=self._feasible(ps,pp,pm,reference)
        accept=psafe[0]&(~joint.any()|pjoint[0])
        best=torch.index_select(candidates,0,scores.argmax().reshape(1)).squeeze(0)
        plan=torch.where(accept,proposal,best)
        return plan,reference,joint.any().float(),safe.any().float()

    def _reference(self,se):
        return torch.where(self.state[5]>0,se[0],self.state[4])

    @torch.no_grad()
    def reset(self,window):
        self.initial=float(np.clip(window[-1,5],self.low,self.high))
        self.completed_se=0.;self.completed_prb=0.;self.reference=1.
        self.step=0
        self.previous=np.full(H,.84*self.initial,dtype=np.float32)

    @torch.no_grad()
    def decide(self,window):
        if self.step>=H:
            raise RuntimeError('Finite eight-step planning episode finished; reset with a new reference before deciding again.')
        self.window.copy_(torch.as_tensor(window[None],device='cuda'))
        self.nominal.copy_(torch.as_tensor(self.previous,device='cuda'))
        self.state.copy_(torch.tensor([self.initial,H-self.step,self.completed_se,self.completed_prb,
                                      self.reference,float(self.step==0)],device='cuda'))
        self.noise.normal_()
        self.graph.replay()
        plan,reference,joint,safe=self.result
        # D2H copies are intentionally part of end-to-end decision timing.
        self.previous=np.concatenate([plan[1:].cpu().numpy(),plan[-1:].cpu().numpy()])
        self.reference=float(reference.cpu())
        action=float(plan[0].cpu())
        return action,dict(joint_candidate=bool(joint.cpu()),safe_candidate=bool(safe.cpu()))

    def observe(self,se,action):
        self.completed_se+=se;self.completed_prb+=action;self.step+=1

def load_optimized_planner():
    """Default entry point for the validation-selected, hard-SE MPPI planner."""
    selected=json.loads((OPT/'selected_mppi.json').read_text())
    return GraphMPPI(load_bundle('L32'),population=selected['population'],
                     sigma=selected['sigma'],temperature=selected['temperature'],enforce_se=True)
