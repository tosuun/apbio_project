# APBIO Semester Project

Course project for Bio-Inspired Machine Learning (APBIO) at Universidad
de Vigo, 2025/26. Author: İbrahim Emir Tosun (Erasmus student).

## What this is

I combined a Modern Hopfield Network with DQN. The Hopfield part replaces
the standard random replay buffer: when the agent samples a mini-batch
for training, the buffer uses the most recent observation as a query and
returns transitions whose states are most similar to it. Plain DQN just
samples uniformly at random.

I tested this on CartPole-v1 from Gymnasium. The full report is in
`report/apbio_project.pdf`.

## What's in the repo

```
notebooks/      Colab notebooks (this is where everything was actually run)
scripts/        plot_architecture.py and plot_results.py
logs/           per-seed CSV logs from every training run
figures/        all the figures from the report
report/         main.tex, references.bib, main.pdf
requirements.txt
```

There are 4 notebooks:

- `01_baseline_dqn.ipynb` — plain DQN sanity check on CartPole
- `02_hopfield_module.ipynb` — Hopfield network on its own (recall tests)
- `03_hopfield_dqn.ipynb` — first hybrid run, single seed
- `04_main_experiment.ipynb` — main experiment + sweeps + best config

## How to reproduce

I ran everything on Google Colab with a CPU runtime, no GPU. To reproduce:

1. Open `notebooks/04_main_experiment.ipynb` in Colab.
2. Mount Drive and run all cells. The notebook installs its own packages.
3. The script writes per-seed CSVs to `logs/` and figures to `figures/`.

If you want to run it locally instead:

```bash
git clone https://github.com/tosuun/apbio_project.git
cd apbio_project
pip install -r requirements.txt
jupyter notebook notebooks/04_main_experiment.ipynb
```

The main experiment (5 seeds × baseline + 5 seeds × hybrid) takes about
30 minutes on a CPU. The full set of sweeps adds another hour.

## Regenerating the figures

If the CSVs in `logs/` are already there, you don't need to retrain to
get the figures back:

```bash
python scripts/plot_results.py
```

This rebuilds every figure in `figures/` from the raw logs.

## Notes

- This is an individual project.
- The hyperparameters are the Stable-Baselines3 RL Zoo defaults for DQN
  on CartPole-v1.
- The bibliography in the report cites the Hopfield paper, the DQN paper,
  Stable-Baselines3, Gymnasium, and Prioritized Experience Replay.
