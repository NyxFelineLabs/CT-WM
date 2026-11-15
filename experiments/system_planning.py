"""Paired receding-horizon screening inside the learned world model only."""
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
from system_common import ROOT, OUT, EXPERIMENTS, load_bundle, core, write_json, sync
from system_prediction import COLUMNS

POLICIES = ['hold', 'step_minus20', 'greedy', 'cem']
SEEDS = [42, 43, 44]
STEPS = 8
STARTS_PER_FILE = 8

@torch.no_grad()
def prior_frame(b, window):
    x = torch.as_tensor(b.x_scaler.transform(window[None]), device=b.cfg.device)
    d = b.model.forward_d(x, x[:,:,5:6])
    z, _ = b.model._split_gauss(b.model.prior_net(d))
    return b.model.full_decoder(torch.cat([d,z], -1))

@torch.no_grad()
def choose(b, window, policy, initial):
    low, high = b.meta['action_bounds']['low'], b.meta['action_bounds']['high']
    if policy == 'cem':
        return core.wm_ms3m_cem_plan_next_prb(b, window)[0]
    if policy == 'hold':
        return float(np.clip(initial, low, high))
    if policy == 'step_minus20':
        return float(np.clip(initial*.8, low, high))
    # Exact H=1 greedy grid under the reference planner's delayed-action semantics.
    # Current y does not depend on candidate a; current reward does include PRB cost.
    frame = prior_frame(b, window)
    grid = torch.linspace(low, high, 256, device=b.cfg.device)
    actions_s = ((grid-b.x_scaler.mean[5])/b.x_scaler.std[5])[:,None]
    prev_s = (window[-1,5]-b.x_scaler.mean[5])/b.x_scaler.std[5]
    reward = b.reward_fn(frame.expand(256,-1), actions_s)[:,0]
    score = reward-b.cfg.action_smooth_penalty*abs(actions_s[:,0]-float(prev_s))
    return float(grid[score.argmax()].cpu())

@torch.no_grad()
def episode(b, environment, original, policy, seed, file, start):
    core.set_seed(seed)
    window = original.copy()
    initial = float(window[-1,5])
    records = []
    train_low = np.array(environment.meta['x_mean']) - 3*np.array(environment.meta['x_std'])
    train_high = np.array(environment.meta['x_mean']) + 3*np.array(environment.meta['x_std'])
    for step in range(STEPS):
        sync(); begin = time.perf_counter()
        action = choose(b, window[-b.meta['L']:], policy, initial)
        sync(); decision_end = time.perf_counter()
        # Model prediction uses history, then chosen action enters the next history.
        # Decode Y using Y statistics, then feedback in physical units and rescale X.
        ys = prior_frame(environment, window[-environment.meta['L']:])
        y = environment.y_full_scaler.inverse_transform(ys.cpu().numpy())[0]
        a_s = torch.tensor([[(action-environment.x_scaler.mean[5])/environment.x_scaler.std[5]]],
                           device=environment.cfg.device, dtype=torch.float32)
        prev_s = (window[-1,5]-environment.x_scaler.mean[5])/environment.x_scaler.std[5]
        reward = float((environment.reward_fn(ys,a_s)-environment.cfg.action_smooth_penalty*abs(a_s-float(prev_s))).cpu())
        assert np.isfinite(y).all() and np.isfinite(reward)
        next_frame = y.copy()
        next_frame[5] = action  # executed PRB command is observed in feedback
        window = np.concatenate([window[1:],next_frame[None]],axis=0)
        sync(); end = time.perf_counter()
        records.append(dict(file=file, target_row=start, seed=seed, policy=policy, step=step+1,
            SINR_db=float(y[9]), SE=float(y[10]), PRB=action, RSRP_dbm=float(y[7]),
            BLER=float(y[11]), Delay=float(y[12]), reward=reward,
            decision_ms=1000*(decision_end-begin), cycle_ms=1000*(end-begin),
            negative_se=bool(y[10]<0), outside_train_3sigma_fraction=float(np.mean((next_frame<train_low)|(next_frame>train_high)))))
    return records

def summarize(df):
    means = ['SINR_db','SE','PRB','RSRP_dbm','BLER','Delay','reward']
    out = []
    for (model, policy), group in df.groupby(['model','policy'],sort=False):
        e = group.groupby(['file','target_row','seed'])[means].mean()
        totals = group.groupby(['file','target_row','seed']).reward.sum()
        out.append(dict(model=model,policy=policy,episodes=len(e),decisions=len(group),
            **{k:float(e[k].mean()) for k in means}, cumulative_reward=float(totals.mean()),
            decision_mean_ms=float(group.decision_ms.mean()),decision_p50_ms=float(group.decision_ms.median()),
            decision_p95_ms=float(group.decision_ms.quantile(.95)),cycle_mean_ms=float(group.cycle_ms.mean()),
            cycle_p95_ms=float(group.cycle_ms.quantile(.95)),
            decision_over_20ms_fraction=float((group.decision_ms>20).mean()),
            decision_over_1000ms_fraction=float((group.decision_ms>1000).mean()),
            negative_se_fraction=float(group.negative_se.mean()),
            outside_train_3sigma_fraction=float(group.outside_train_3sigma_fraction.mean())))
    return out

def bootstrap_difference(df):
    # Average seeds first; resample complete start windows, never individual steps.
    rng = np.random.default_rng(20261111)
    ep = df.groupby(['model','policy','file','target_row'])[['SINR_db','SE','PRB','reward']].mean()
    results = []
    for model in EXPERIMENTS:
        base = ep.loc[(model,'hold')]
        for policy in POLICIES[1:]:
            diff = ep.loc[(model,policy)]-base
            arr = diff.to_numpy()
            bs = arr[rng.integers(0,len(arr),size=(2000,len(arr)))].mean(1)
            ci = np.percentile(bs,[2.5,97.5],axis=0)
            means = diff.mean()
            results.append(dict(model=model,policy=policy,
                delta_sinr_db=float(means.SINR_db), delta_sinr_ci=ci[:,0].tolist(),
                delta_se=float(means.SE),delta_se_ci=ci[:,1].tolist(),
                se_gain_percent=float(100*means.SE/base.SE.mean()) if abs(base.SE.mean())>1e-9 else None,
                prb_reduction_percent=float(-100*means.PRB/base.PRB.mean()),delta_prb_ci=ci[:,2].tolist(),
                delta_reward_per_step=float(means.reward),delta_reward_ci=ci[:,3].tolist(),
                note='Exploratory paired window bootstrap; only two held-out files, not a field-trial CI.'))
    return results

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',choices=list(EXPERIMENTS),help='Run one model; default runs all four')
    parser.add_argument('--reuse-completed',action='store_true',
                        help='Explicitly reuse completed CSVs; only for unchanged checkpoints/config.')
    args=parser.parse_args()
    paths = sorted((ROOT / 'data/raw').glob('*.csv'))[-2:]
    frames = [(p.name,pd.read_csv(p)[COLUMNS].to_numpy(np.float32)) for p in paths]
    environment = load_bundle('L32')
    all_records = []
    factual = []
    for name in ([args.name] if args.name else EXPERIMENTS):
        b = load_bundle(name)
        assert b.meta['action_bounds']==environment.meta['action_bounds'], 'Unequal physical action ranges'
        # Factual multistep forecast diagnostic: recorded future controls only,
        # never future KPI feedback; distinct from counterfactual policy gains.
        for file, frame in frames:
            for start in np.linspace(64,len(frame)-STEPS,STARTS_PER_FILE,dtype=int):
                window=frame[start-b.meta['L']:start].copy()
                for h in range(STEPS):
                    ys=prior_frame(b,window)
                    y=b.y_full_scaler.inverse_transform(ys.cpu().numpy())[0]
                    for idx,kpi in [(7,'RSRP'),(9,'SINR'),(10,'SE')]:
                        factual.append(dict(model=name,file=file,target_row=int(start),horizon=h+1,
                                            kpi=kpi,prediction=float(y[idx]),truth=float(frame[start+h,idx])))
                    y[5]=frame[start+h,5]
                    window=np.concatenate([window[1:],y[None]],axis=0)
        pd.DataFrame(factual).to_csv(OUT/'factual_rollout.csv',index=False)
        dest = OUT/f'planning_{name}.csv'
        if args.reuse_completed and dest.exists():
            cached = pd.read_csv(dest)
            assert len(cached)==2*STARTS_PER_FILE*len(SEEDS)*STEPS*len(POLICIES)
            all_records += cached.to_dict('records')
            continue
        assert (b.cfg.plan_horizon,b.cfg.cem_popsize,b.cfg.cem_iters,b.cfg.cem_elite_frac)==(8,256,4,.1)
        # Excluded warm-up of planner, prior transition, and greedy kernels.
        initial = frames[0][1][64-b.meta['L']:64]
        for _ in range(2):
            choose(b,initial,'cem',float(initial[-1,5])); prior_frame(b,initial)
            choose(b,initial,'greedy',float(initial[-1,5]))
        records = []
        for file, frame in frames:
            starts = np.linspace(64,len(frame)-STEPS,STARTS_PER_FILE,dtype=int)
            for start in starts:
                window = frame[start-64:start]
                for seed in SEEDS:
                    for policy in POLICIES:
                        rows = episode(b,environment,window,policy,seed,file,int(start))
                        records += [dict(model=name,**r) for r in rows]
                print('completed', name,file,'start',start,flush=True)
                pd.DataFrame(records).to_csv(OUT/f'planning_{name}_partial.csv',index=False)
        pd.DataFrame(records).to_csv(dest,index=False)
        all_records += records
        print('model finished',name,flush=True)
    df = pd.DataFrame(all_records)
    df.to_csv(OUT/'planning_steps.csv',index=False)
    summary, deltas = summarize(df), bootstrap_difference(df)
    pd.DataFrame(summary).to_csv(OUT/'planning_summary.csv',index=False)
    write_json(OUT/'planning.json',dict(summary=summary,paired_vs_hold=deltas,
        horizon=8,population=256,iterations=4,elite_fraction=.1,
        episodes_per_model_policy=48,control_steps_per_episode=8,
        starts_per_test_file=8,planner_seeds=SEEDS,
        scope='All policies use the same frozen L32 learned-model environment; not a real network or independent simulator.',
        environment='Frozen L32 checkpoint for every planning model and policy; common 64-row history, planner uses its own L.',
        feedback='Y inverse scaling to physical units; next PRB channel overwritten by executed command; X restandardized.',
        reference_planner_unchanged=True, action_effect_delayed_one_transition=True,
        planner_internal_x_y_scaling_mismatch_unmodified=True))
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
