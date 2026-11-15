"""Frozen deployment executor: caches input-independent SSM kernels only."""
import copy
from pathlib import Path
import torch
from torch import nn
from system_common import ROOT, load_bundle, sync, write_json

OPT = ROOT / 'runs/planner_optimization'
OPT.mkdir(parents=True,exist_ok=True)
from runtime_paths import seed_config
seed_config('planning/selected_mppi.json', OPT / 'selected_mppi.json')

class CachedKernel(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.register_buffer('value', value.detach().clone())
    def forward(self):
        return self.value

class PriorMean(nn.Module):
    def __init__(self, bundle):
        super().__init__()
        self.model=copy.deepcopy(bundle.model).eval().requires_grad_(False)
        with torch.no_grad():
            for blk in self.model.blocks:
                blk.kernel=CachedKernel(blk.kernel())
        for k in ['x_mean','x_std','y_full_mean','y_full_std']:
            self.register_buffer(k,torch.tensor(bundle.meta[k],device=bundle.cfg.device))
    def forward(self, physical_window):
        x=(physical_window-self.x_mean)/self.x_std
        d=self.model.forward_d(x,x[:,:,5:6])
        z,_=self.model._split_gauss(self.model.prior_net(d))
        ys=self.model.full_decoder(torch.cat([d,z],-1))
        return ys*self.y_full_std+self.y_full_mean

def make_executor(bundle):
    return PriorMean(bundle).eval()

@torch.no_grad()
def verify(bundle, executor, windows):
    errors=[]
    for window in windows:
        x=torch.as_tensor(bundle.x_scaler.transform(window[None]),device=bundle.cfg.device)
        d=bundle.model.forward_d(x,x[:,:,5:6])
        z,_=bundle.model._split_gauss(bundle.model.prior_net(d))
        reference=bundle.model.full_decoder(torch.cat([d,z],-1))
        physical=reference*executor.y_full_std+executor.y_full_mean
        actual=executor(torch.as_tensor(window[None],device=bundle.cfg.device))
        errors.append(float((actual-physical).abs().max().cpu()))
        torch.testing.assert_close(actual,physical,rtol=2e-5,atol=2e-5)
    return dict(max_absolute_physical_output_error=max(errors),windows=len(errors),
                cached_only_input_independent_kernels=True,world_model_weights_frozen=True)
