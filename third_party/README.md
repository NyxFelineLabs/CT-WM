# Upstream attribution

The multi-scale SSM, latent world-model heads, base training objective and CEM
implementation come from [WM–MS³M](https://github.com/frezazadeh/agentic-wm_ms3m),
based on revision `29e862d9484ee237eecf669247f29e5db76231f1`.
Its [MIT license](WM-MS3M-LICENSE) is retained verbatim.

When using this prediction backbone, cite:

Rezazadeh et al., “Agentic World Modeling for 6G: Near-Real-Time Generative
State-Space Reasoning,” [arXiv:2511.02748](https://arxiv.org/abs/2511.02748).

`src/wm_ms3m/core.py` is copied byte-for-byte from the validated local source.
The local reproduction applied one compatibility repair: replacing
`torch.amp.GradScaler('cuda', ...)` with `torch.cuda.amp.GradScaler(...)`
for the installed PyTorch 2.1 API. No mathematical model or training objective
was changed for this release.
The source hash and the locally applied upstream diff are recorded here so
the upstream base is not mistaken for a claim of pristine source identity.

CT-WM's adapters supply file-isolated splits, portable artifact paths, interval
evaluation, reduced-population planning and a Python link abstraction.
These changes are separate from the original backbone. No ns-3, Docker RIC,
C++ adapter, or unrelated cloned project is bundled.
