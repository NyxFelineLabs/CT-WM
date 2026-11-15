"""Fit variance temperatures on validation only; reuse untouched test samples."""
import hashlib
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar,brentq
from scipy.special import ndtr,logsumexp
from planner_runtime import OPT
from system_common import ROOT,load_bundle,core,write_json
from system_prediction import COLUMNS,quantile

def coverage(mu,var,y,T,level):
    pit=ndtr((y[:,None]-mu)/np.sqrt(var*T)).mean(1)
    return float(np.mean((pit>=(1-level)/2)&(pit<=(1+level)/2)))

def nll(mu,var,y,T):
    v=var*T
    return float(-np.mean(logsumexp(-.5*((y[:,None]-mu)**2/v+np.log(2*np.pi*v)),axis=1)-np.log(8)))

@torch.no_grad()
def collect_validation():
    b=load_bundle('L32');core.set_seed(20261111)
    p=sorted((ROOT/'data/raw').glob('*.csv'))[7]
    raw=pd.read_csv(p)[COLUMNS].to_numpy(np.float32)
    windows=np.lib.stride_tricks.sliding_window_view(raw,32,axis=0)[32:-1].transpose(0,2,1).copy()
    means=[];variances=[]
    for start in range(0,len(windows),256):
        x=torch.as_tensor(b.x_scaler.transform(windows[start:start+256]),device='cuda')
        m=[];v=[]
        for _ in range(8):
            (mu,lv),_,_=b.model(x,x[:,:,5:6],use_posterior=False)
            m.append(mu[:,0].cpu().numpy().astype(np.float64)*b.y_scaler.std[0]+b.y_scaler.mean[0])
            v.append(lv[:,0].exp().cpu().numpy().astype(np.float64)*b.y_scaler.std[0]**2)
        means.append(np.stack(m,1));variances.append(np.stack(v,1))
    result=dict(mu=np.concatenate(means),var=np.concatenate(variances),y=raw[64:,7].astype(np.float64))
    assert len(result['y'])==len(result['mu'])
    np.savez_compressed(OPT/'calibration_validation.npz',**result)
    return result,p

def evaluate(data,temperatures,split,intervals):
    mu,var,y=data['mu'],data['var'],data['y']
    rows=[]
    for method,levels in temperatures.items():
        for level,T in levels.items():
            lo=quantile(mu,np.sqrt(var*T),(1-level)/2)
            hi=quantile(mu,np.sqrt(var*T),(1+level)/2)
            cov=float(np.mean((y>=lo)&(y<=hi)))
            intervals[f'{split}_{method}_{int(level*100)}']=np.stack([lo,hi],1)
            rows.append(dict(split=split,method=method,nominal=level,temperature=T,coverage=cov,
                             error_pp=100*(cov-level),mean_width_db=float(np.mean(hi-lo)),
                             nll_nats=None if method=='levelwise_TS' else nll(mu,var,y,T),
                             file_coverage=None if 'file_id' not in data else [
                                 float(np.mean(((y>=lo)&(y<=hi))[data['file_id']==i])) for i in range(2)]))
    return rows

def main():
    val,path=collect_validation()
    mu,var,y=val['mu'],val['var'],val['y']
    nll_fit=minimize_scalar(lambda logT:nll(mu,var,y,np.exp(logT)),bounds=(-8,8),method='bounded')
    nll_T=float(np.exp(nll_fit.x))
    grid=np.exp(np.linspace(-6,4,401))
    balanced_T=float(min(grid,key=lambda T:sum((coverage(mu,var,y,T,l)-l)**2 for l in [.5,.95])))
    level_T={}
    for level in [.5,.95]:
        root=brentq(lambda logT:coverage(mu,var,y,np.exp(logT),level)-level,-10,10,xtol=1e-7)
        level_T[level]=float(np.exp(root))
    temps={'uncalibrated':{.5:1.,.95:1.},'scalar_TS_NLL':{.5:nll_T,.95:nll_T},
           'scalar_TS_coverage':{.5:balanced_T,.95:balanced_T},'levelwise_TS':level_T}
    # Freeze choices before opening the held-out test artifact.
    config=dict(fit_split='validation CSV #8 only, raw target rows >=64',
        validation_file=path.name,validation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        validation_n=len(y),mc_samples=8,temperatures=temps,
        mean_unchanged=True,scaling='component conditional variance multiplied by T',
        levelwise_note='Two interval-level temperatures; not a single Gaussian variance or parameter posterior.')
    write_json(OPT/'calibration_frozen_config.json',config)
    intervals={}
    rows=evaluate(val,temps,'validation',intervals)
    old=np.load(ROOT/'runs/system_validation/prediction_L32.npz')
    test=dict(mu=old['component_mu'],var=old['component_var'],y=old['true'][:,7].astype(np.float64),file_id=old['file_id'])
    rows+=evaluate(test,temps,'test',intervals)
    for split in ['validation','test']:
        inner,outer=intervals[f'{split}_levelwise_TS_50'],intervals[f'{split}_levelwise_TS_95']
        assert (outer[:,0]<=inner[:,0]).all() and (outer[:,1]>=inner[:,1]).all()
    np.savez_compressed(OPT/'calibration_intervals.npz',**intervals,
                        test_true=test['y'],test_file_id=test['file_id'],test_mean=test['mu'].mean(1))
    # Exact same component means and test predictions as archived evaluation.
    write_json(OPT/'calibration.json',dict(config=config,rows=rows,
        test_n=len(test['y']),no_test_fitting=True,test_component_means_unchanged=True))
    pd.DataFrame(rows).to_csv(OPT/'calibration_summary.csv',index=False)
    print('FROZEN TEMPERATURES',temps,flush=True)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':main()
