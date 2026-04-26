# APBIO Semester Project

Course project for **Bio-Inspired Machine Learning (APBIO)** at Universidad de
Vigo, 2025/26.

The goal is to combine a bio-inspired ML component with a reinforcement learning
algorithm. My choice:

- **Bio-inspired component:** Modern Hopfield Network, used as a replay buffer.
- **RL algorithm:** DQN.
- **Environment:** CartPole-v1 (Gymnasium).

Instead of sampling past experiences uniformly at random (which is what plain
DQN does), the agent uses the current state as a query to a Hopfield
associative memory, and trains on the transitions that are most similar to
that query. The idea is that this should give the agent a more informative
mini-batch than random sampling.

## How to run

I use Google Colab for training. The notebooks live in `notebooks/` and each
one installs its own dependencies in the first cell.

For a local setup:

```bash
git clone https://github.com/tosuun/apbio_project.git
cd apbio_project
pip install -r requirements.txt
```

The main experiment (5 seeds, baseline + hybrid) can be reproduced with:

```bash
python -m src.train --mode baseline
python -m src.train --mode hybrid
python scripts/plot_results.py
```

Expected runtime: ~90 minutes on a laptop CPU (no GPU needed).

## Repo layout

```
notebooks/   Colab notebooks where most of the work happens
src/         Python modules (Hopfield buffer, training loop, utils)
configs/     YAML config files
logs/        Per-seed CSV logs
figures/     Figures used in the report
report/      LaTeX source of the written report
```

## Status

- [x] Baseline DQN running on CartPole
- [ ] Hopfield replay buffer module
- [ ] Hybrid integration
- [ ] Hyperparameter sweeps
- [ ] Ablation study
- [ ] Final report
- [ ] Video presentation

## Notes

Individual project. All code and writing are my own. Dependencies
(Stable-Baselines3, Gymnasium, PyTorch) are cited in the report.
