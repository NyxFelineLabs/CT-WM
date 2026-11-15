"""Pure Python E2 service-model semantics simulator, NOT ASN.1 E2SM/E2AP.

Dataclasses/JSON model the KPM -> xApp -> RC application loop. The link model
is an explicitly assumed flat-fading logistic link abstraction, not ns-3.
"""
from dataclasses import dataclass,asdict
from collections import deque
import json
import traceback
import numpy as np
from ct_common import ROOT,frames,starts,sha,write_json
from simple_olla import SimpleOLLA

OUT=ROOT/'runs/e2_python'
OUT.mkdir(parents=True,exist_ok=True)
from runtime_paths import seed_config
for _name in ['planner_config.json', 'forecast_validation.json', 'reward_validation.json']:
    seed_config('e2/' + _name, OUT / _name)
ETA=np.array([.15,.19,.23,.31,.38,.49,.6,.74,.88,1.03,1.18,1.33,1.48,1.7,1.91,2.16,
    2.41,2.57,2.73,3.03,3.32,3.61,3.9,4.21,4.52,4.82,5.12,5.33,5.55])
BASE=dict(horizon=8,steps=128,seeds=[42,43,44],starts_per_file=8,prb_bounds=[13,25],mcs_bounds=[0,28],
    target_bler=.1,nack_step_db=.9,ack_step_db=.1,offset_sign='SINR minus offset',default_mcs=14,
    subframes_per_cycle=20,cycle_ms=20,total_prb=25,
    shannon_gap_db=3.,logistic_slope_db=1.,resource_elements_per_RB=168,payload_fraction=.75,
    power_model='Fixed UE total power: per-RB SINR gain=10log10(initial_PRB/current_PRB)',
    population=32,temperature=.15,safety_floor=.95,quality_target=.98,smooth_weight=.02,
    budget_ms=15.,rl_steps_each=50000,training_seed=42,T50=.154337508,T95=1.591854689,
    action_space='[PRB_adjustment, absolute MCS_index]',
    service_model='Semantic local KPM/RC dataclasses with JSON roundtrip, no ASN.1/E2AP/SCTP/RIC containers',
    link_model='Assumed Shannon-gap thresholds + logistic first-transmission BLER; not calibrated real PHY',
    reference='A: standard OLLA + fixed initial PRB; paired exogenous SINR and common TB random numbers',
    blind_holdout=False,model_weights_frozen=True)

def guard(fn):
    try:fn()
    except Exception:
        with (OUT/'errors_full.log').open('a',encoding='utf-8') as f:traceback.print_exc(file=f)
        raise

@dataclass(frozen=True)
class E2SMKPMIndication:
    node_id:str
    ue_id:int
    sequence:int
    timestamp_ms:int
    cqi:int
    sinr_db:float
    ack_nack:tuple
    current_prb:int
    current_mcs:int
    kpis:tuple
    def __post_init__(self):
        if len(self.kpis)!=13 or not np.isfinite(self.kpis).all():raise ValueError('KPM must carry 13 finite KPI values')
        if not 0<=self.cqi<=15 or not 13<=self.current_prb<=25 or not 0<=self.current_mcs<=28:raise ValueError('KPM range violation')
        if any(type(a) is not bool for a in self.ack_nack):raise TypeError('ACK/NACK requires bool; no aggregate-BLER pseudo feedback')
    def wire(self):return json.dumps(asdict(self),allow_nan=False,separators=(',',':'))
    @classmethod
    def read(cls,s):
        v=json.loads(s);v['ack_nack']=tuple(v['ack_nack']);v['kpis']=tuple(v['kpis']);return cls(**v)

@dataclass(frozen=True)
class E2SMRCControl:
    node_id:str
    ue_id:int
    indication_sequence:int
    request_id:int
    prb_adjustment:int
    mcs_index:int
    def __post_init__(self):
        if type(self.prb_adjustment) is not int or type(self.mcs_index) is not int:raise TypeError('RC actions must be integer')
        if not 0<=self.mcs_index<=28:raise ValueError('MCS index outside 0..28')
    def wire(self):return json.dumps(asdict(self),allow_nan=False,separators=(',',':'))
    @classmethod
    def read(cls,s):return cls(**json.loads(s))

class PythonLink:
    def __init__(self):
        self.eta=ETA.copy();self.threshold10=10*np.log10(np.expm1(ETA*np.log(2)))+BASE['shannon_gap_db']
        p=np.arange(13,26)[:,None];self.bits=(np.floor(p*168*.75*ETA[None,:]/8)*8).astype(np.int32)
    def effective(self,channel,p,hold):return np.asarray(channel)+10*np.log10(np.asarray(hold)/np.asarray(p))
    def bler(self,channel,p,m,hold):
        effective=self.effective(channel,p,hold);mid=self.threshold10[np.asarray(m,dtype=int)]-np.log(9.)
        return 1/(1+np.exp(np.clip((effective-mid)/BASE['logistic_slope_db'],-60,60)))
    def se(self,channel,p,m,hold):
        b=self.bler(channel,p,m,hold)
        return self.bits[np.asarray(p,dtype=int)-13,np.asarray(m,dtype=int)]*(1-b)/4500.,b
    def select(self,channel,p,hold,offset=0.):
        bl=self.bler(np.asarray(channel)[...,None]-np.asarray(offset)[...,None],np.asarray(p)[...,None],np.arange(29),np.asarray(hold)[...,None])
        return np.where(bl<=.1,np.arange(29),0).max(-1)
    def cqi(self,sinr):
        efficiencies=np.array([0,.15,.23,.38,.6,.88,1.18,1.48,1.91,2.41,2.73,3.32,3.9,4.52,5.12,5.55])
        cap=np.log2(1+10**((np.asarray(sinr)-3)/10))
        return np.clip(np.searchsorted(efficiencies,cap,side='right')-1,0,15)

def olla():return SimpleOLLA()

class VirtualE2Node:
    def __init__(self,link):self.link=link
    def indication(self,row,sequence,acks=()):
        return E2SMKPMIndication('sim-e2-001',1,int(sequence),int(sequence*20),int(self.link.cqi(row[9])),
            float(row[9]),tuple(bool(a) for a in acks),int(row[5]),int(row[0]),tuple(float(v) for v in row))
    def reset(self,frame,start,seed):
        self.frame=frame;self.start=int(start);self.t=0;self.last_request=-1
        hist=frame[start-32:start].copy();hist[:,0]=np.clip(np.rint(hist[:,0]),0,28);hist[:,5]=np.clip(np.rint(hist[:,5]),13,25)
        self.hold=int(hist[-1,5]);self.fixed_mcs=BASE['default_mcs']
        hist[-1,0]=self.fixed_mcs
        hist[:,9]+=10*np.log10(self.hold/hist[:,5]);hist[:,1]=self.link.cqi(hist[:,9]);self.current=hist[-1].copy()
        self.reference_olla=olla();self.uniforms=np.random.default_rng(seed).random((128,20))
        self.bootstrap=[self.indication(r,i-31).wire() for i,r in enumerate(hist)]
        self.last_indication=E2SMKPMIndication.read(self.bootstrap[-1]);return self.bootstrap
    def execute(self,control):
        if control.node_id!='sim-e2-001' or control.ue_id!=1:raise ValueError('RC node/UE mismatch')
        if control.indication_sequence!=self.t or control.request_id<=self.last_request:raise ValueError('Stale/replayed RC request')
        self.last_request=control.request_id
        old=self.current.copy();p=int(np.clip(old[5]+control.prb_adjustment,13,25));m=control.mcs_index
        # Reference uses its own counterfactual ACKs; never exposed to xApps.
        old_channel=float(old[9]-10*np.log10(self.hold/old[5]))
        rm=int(self.link.select(old_channel,self.hold,self.hold,self.reference_olla.offset_db))
        row=self.frame[self.start+self.t].copy();channel=float(row[9]);u=self.uniforms[self.t]
        expected,b=self.link.se(channel,p,m,self.hold);r_expected,rb=self.link.se(channel,self.hold,rm,self.hold)
        acks=u>=b;racks=u>=rb
        for a in racks:self.reference_olla.update(bool(a))
        se=float(self.link.bits[p-13,m]*acks.mean()/4500.);ref=float(self.link.bits[self.hold-13,rm]*racks.mean()/4500.)
        row[0]=m;row[1]=self.link.cqi(self.link.effective(channel,p,self.hold));row[5]=p
        row[9]=self.link.effective(channel,p,self.hold);row[10]=se;row[11]=1-acks.mean()
        self.current=row;self.t+=1;self.last_indication=self.indication(row,self.t,acks)
        metrics=dict(SE=se,reference_SE=ref,expected_SE=float(expected),reference_expected_SE=float(r_expected),PRB=p,MCS=m,
            reference_PRB=self.hold,reference_MCS=rm,SINR=float(row[9]),channel_SINR=channel,BLER=float(1-acks.mean()),
            predicted_BLER=float(b),ACK_count=int(acks.sum()),NACK_count=int((~acks).sum()),
            previous_PRB=int(old[5]),previous_MCS=int(old[0]),abs_delta_PRB=abs(p-old[5]),abs_delta_MCS=abs(m-old[0]))
        return self.last_indication.wire(),metrics

class XAppContext:
    def __init__(self,link):self.link=link
    def reset(self,bootstrap):
        self.history=deque(maxlen=32);self.offset=olla();self.last_sequence=-100;self.request=0
        for wire in bootstrap:
            k=E2SMKPMIndication.read(wire);self.history.append(k.kpis)
        self.current=k;self.last_sequence=k.sequence;self.hold=k.current_prb;self.fixed_mcs=BASE['default_mcs']
    def receive(self,kpm):
        if kpm.sequence<self.last_sequence:raise ValueError('KPM out of order')
        if kpm.sequence>self.last_sequence:
            if kpm.sequence!=self.last_sequence+1:raise ValueError('KPM sequence gap')
            for a in kpm.ack_nack:self.offset.update(a)
            self.history.append(kpm.kpis);self.last_sequence=kpm.sequence
        self.current=kpm
    def window(self):return np.array(self.history,dtype=np.float32)
    def channel(self):return self.current.sinr_db-10*np.log10(self.hold/self.current.current_prb)
    def fallback(self):return self.hold,int(self.link.select(self.channel(),self.hold,self.hold,self.offset.offset_db))
    def control(self,p,m):
        self.request+=1;k=self.current
        return E2SMRCControl(k.node_id,k.ue_id,k.sequence,self.request,int(p)-k.current_prb,int(m))

def score(se,ref,p,hold,dp,dm,config):
    scale=np.maximum(np.abs(ref),.05)
    return se/scale-config['penalty']*np.maximum(.98*ref-se,0)/scale+config['resource']*(1-p/hold)-.02*(np.abs(dp)/2+np.abs(dm)/2)

def freeze():
    write_json(OUT/'protocol.json',dict(BASE,core_sha256=sha(ROOT/'src/wm_ms3m/core.py'),
        checkpoint_sha256=sha(ROOT/'artifacts/file_split_width_93/best.pt'),
        data_files={s:[n for n,_ in frames(s)] for s in ['train','val','test']},link_eta=ETA.tolist(),
        installation='Existing Python/PyTorch environment, no C++ binding, native library or container call.'))

if __name__=='__main__':guard(freeze)
