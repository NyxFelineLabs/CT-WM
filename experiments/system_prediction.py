"""Common-target prediction and exact eight-component mixture calibration."""
import json
import math
import time
import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr, logsumexp
from system_common import ROOT, OUT, EXPERIMENTS, load_bundle, core, write_json, sync

COLUMNS = ['phy_mcs', 'mac_dl_cqi', 'mac_dl_ri', 'mac_dl_pmi', 'mac_ul_buffer',
           'mac_n_prb', 'rsrq', 'rsrp', 'rssi', 'dl_sinr', 'se', 'dl_bler', 'delay']

def metrics(pred, true):
    err = pred.astype(np.float64) - true.astype(np.float64)
    mse = float(np.mean(err**2))
    return dict(mae=float(np.mean(abs(err))), rmse=math.sqrt(mse), mse=mse,
                r2=float(1 - np.sum(err**2)/np.sum((true-true.mean())**2)))

def quantile(mu, std, q):
    lo = (mu - 12*std).min(1)
    hi = (mu + 12*std).max(1)
    for _ in range(45):
        mid = (lo + hi)/2
        cdf = ndtr((mid[:, None] - mu)/std).mean(1)
        lo = np.where(cdf < q, mid, lo)
        hi = np.where(cdf >= q, mid, hi)
    return (lo + hi)/2

@torch.no_grad()
def main():
    frames = [pd.read_csv(p)[COLUMNS].to_numpy(np.float32)
              for p in sorted((ROOT / 'data/raw').glob('*.csv'))[-2:]]
    truth = np.concatenate([f[64:] for f in frames])
    file_ids = np.concatenate([np.full(len(f)-64, i, np.int64) for i, f in enumerate(frames)])
    last = np.concatenate([f[63:-1] for f in frames])
    results = {'common_test_size': len(truth), 'common_target_first_row_zero_based': 64,
               'mc_samples': 8, 'seed': 20261111, 'models': {},
               'persistence_rsrp': metrics(last[:, 7], truth[:, 7])}
    for name in EXPERIMENTS:
        b = load_bundle(name)
        core.set_seed(20261111)
        L = b.meta['L']
        windows = np.concatenate([np.lib.stride_tricks.sliding_window_view(f, L, axis=0)
                                  [64-L:-1].transpose(0,2,1) for f in frames]).copy()
        assert len(windows) == len(truth)
        mus, variances, fulls = [], [], []
        for start in range(0, len(windows), 256):
            x = torch.as_tensor(b.x_scaler.transform(windows[start:start+256]), device=b.cfg.device)
            a = x[:, :, 5:6]
            means, vars_, frames_ = [], [], []
            for _ in range(8):
                (mu, lv), full, _stats = b.model(x, a, use_posterior=False)
                means.append(mu[:, 0].cpu().numpy().astype(np.float64)*b.y_scaler.std[0]+b.y_scaler.mean[0])
                vars_.append(lv[:, 0].exp().cpu().numpy().astype(np.float64)*b.y_scaler.std[0]**2)
                frames_.append(b.y_full_scaler.inverse_transform(full.cpu().numpy()))
            mus.append(np.stack(means, 1)); variances.append(np.stack(vars_, 1))
            fulls.append(np.mean(frames_, axis=0))
        mu, var, full = np.concatenate(mus), np.concatenate(variances), np.concatenate(fulls)
        assert np.isfinite(mu).all() and np.isfinite(var).all() and (var > 0).all()
        std, y = np.sqrt(var), truth[:, 7].astype(np.float64)
        aleatoric = var.mean(1)
        latent = mu.var(1, ddof=0)
        calibration = []
        bounds = {}
        for level in (.50, .80, .90, .95):
            lo, hi = quantile(mu, std, (1-level)/2), quantile(mu, std, (1+level)/2)
            covered = (y >= lo) & (y <= hi)
            calibration.append(dict(nominal=level, coverage=float(covered.mean()),
                                    coverage_error=float(covered.mean()-level),
                                    mean_width_db=float((hi-lo).mean()),
                                    file_coverage=[float(covered[file_ids == i].mean()) for i in range(2)]))
            bounds[str(level)] = np.stack([lo, hi], 1)
        logpdf = -.5*((y[:,None]-mu)**2/var + np.log(2*np.pi*var))
        nll = float(-np.mean(logsumexp(logpdf, axis=1) - np.log(8)))
        pit = ndtr((y[:,None]-mu)/std).mean(1)
        target_metrics = metrics(mu.mean(1), y)
        per_kpi = {k: metrics(full[:,i], truth[:,i]) for i,k in enumerate(b.cfg.feature_names)}
        item = dict(parameters=sum(p.numel() for p in b.model.parameters()), length=L,
                    target_rsrp=target_metrics, full_decoder_kpis=per_kpi,
                    files=[metrics(mu[file_ids==i].mean(1), y[file_ids==i]) for i in range(2)],
                    mean_aleatoric_variance_db2=float(aleatoric.mean()),
                    mean_latent_mean_variance_db2=float(latent.mean()),
                    mean_total_variance_db2=float((aleatoric+latent).mean()),
                    epistemic_parameter_variance=None,
                    note='Latent mean variance is NOT parameter epistemic uncertainty.',
                    nll_nats=nll, calibration=calibration,
                    mean_absolute_coverage_error=float(np.mean([abs(c['coverage_error']) for c in calibration])),
                    pit_histogram=np.histogram(pit, bins=np.linspace(0,1,11))[0].tolist())
        xb = torch.as_tensor(b.x_scaler.transform(windows[:256]),device=b.cfg.device)
        ab = xb[:,:,5:6]
        def predict_eight():
            acc = 0.
            for _ in range(8):
                (m,_),_,_ = b.model(xb,ab,use_posterior=False)
                acc = acc + m
            return acc/8
        for _ in range(10):
            predict_eight()
        timings=[]
        for _ in range(100):
            sync(); t0=time.perf_counter(); predict_eight(); sync()
            timings.append(1000*(time.perf_counter()-t0))
        item['latency_mc8_batch256'] = dict(mean_ms=float(np.mean(timings)),
            per_sample_ms=float(np.mean(timings)/256),p95_ms=float(np.percentile(timings,95)),
            raw_batch_ms=timings,warmup=10,repeats=100)
        results['models'][name] = item
        np.savez_compressed(OUT / f'prediction_{name}.npz', component_mu=mu, component_var=var,
                            true=truth, full_prediction=full, file_id=file_ids, pit=pit,
                            **{f'interval_{k}':v for k,v in bounds.items()})
        write_json(OUT / 'prediction.json', results)
        print(name, json.dumps(item, ensure_ascii=False), flush=True)
    pd.DataFrame([dict(model=k, L=v['length'], parameters=v['parameters'], **v['target_rsrp'])
                  for k,v in results['models'].items()]).to_csv(OUT/'prediction_summary.csv', index=False)

if __name__ == '__main__':
    main()
