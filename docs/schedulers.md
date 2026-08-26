# Schedulers

| Name              | Venue | Year | Paper | URL |
| ----------------- | ----- | ---- | ----- | --- |
| BaseScheduler     |||||
| AutoCyclic        | IEEE Access | 2024 | AutoCyclic: Deep Learning Optimizer for Time Series Data Prediction | [IEEEXplore](https://ieeexplore.ieee.org/document/10410839) - [GITHUB](https://github.com/wtfish/AutoCyclic) |
| CAWR              | ICLR | 2017 | SGDR: Stochastic Gradient Descent with Warm Restarts | [Arxiv](https://arxiv.org/abs/1608.03983) |
| CosineAnnealingLR |||||
| ExpHyperbolicLR   ||| HyperbolicLR: Epoch Insensitive Learning Rate Scheduler | [Arxiv](https://arxiv.org/abs/2407.15200) - [GITHUB](https://github.com/Axect/HyperbolicLR) |
| HyperbolicLR      ||| HyperbolicLR: Epoch Insensitive Learning Rate Scheduler | [Arxiv](https://arxiv.org/abs/2407.15200) - [GITHUB](https://github.com/Axect/HyperbolicLR) |
| OneCycleLR        | arXiv | 2017 | Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates | [Arxiv](https://arxiv.org/abs/1708.07120) - [PYTORCH](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html) |
| StepLR            |||||

## Lifecycle modes

`--scheduler_mode` controls scheduler lifecycle independently from its learning-rate formula:

- `batch`: restart each federated iteration and step after every optimizer update.
- `epoch`: restart each federated iteration and step after every local epoch.
- `iteration`: preserve per-client state across federated iterations and step after every local epoch.

The default is `iteration`. `AutoCyclic` and `OneCycleLR` compel `batch`; other schedulers allow all three modes. Batch and epoch horizons use the current local loader and epoch count, while iteration uses `iterations * epochs`.

For both hyperbolic schedulers, the paper's `N` is the selected horizon minus one; `U` is `--upper_bound` times that horizon. Set `three_phase=True` when reproducing the OneCycle paper's three-phase schedule.
