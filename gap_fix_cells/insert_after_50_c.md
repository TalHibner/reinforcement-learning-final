# Reload from Checkpoints (optional — skip training entirely)

Every training cell writes its logger (`artifacts/logs/*.json`), its final and
best-so-far weights (`artifacts/checkpoints/*.pt`), its figures
(`artifacts/plots/*.png`) and its videos (`videos/*.mp4`) to `STORAGE_ROOT` as it
goes. This section is the inverse operation: it rebuilds every agent and logger
from those artifacts, so the notebook can be run **top-to-bottom in minutes
without training anything** — which is the intended way to review it.

**How to use it:** set `LOAD_FROM_ARTIFACTS = True` in the cell below and run it.
It reconstructs `dqn_simple_agent`, `reinforce_simple_agent`, `ppo_simple_agent`,
`dqn_complex_agent`, `reinforce_complex_agent`, `ppo_complex_agent` and their six
loggers, using exactly the architectures implied by `BEST_SETTINGS`. Then skip the
training cells and run the plotting / evaluation / video cells directly. Left at
`False`, this cell does nothing and training proceeds normally.

Which weights are loaded is stated explicitly: `prefer_best=True` (the default)
loads the best-so-far checkpoint — the same model-selection rule used for the
reported results — and falls back to the final weights when no best checkpoint
exists on disk.
