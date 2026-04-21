# APBIO Semester Project — Bio-Inspired Hybrid System with Reinforcement Learning

**Course:** Bio-Inspired Machine Learning (APBIO) — Universidad de Vigo
**Academic year:** 2025–2026
**Author:** *tosuun*

---

## 1. Overview

This repository contains a **hybrid reinforcement learning agent** that combines:

- a **bio-inspired component**: a *Modern Hopfield Network* used as a content-addressable replay buffer, and
- a **reinforcement learning algorithm**: *Deep Q-Network (DQN)*.

Instead of sampling past transitions uniformly at random (as standard DQN does), our agent queries a Hopfield associative memory with the current state and retrieves a mini-batch of structurally similar past experiences. The DQN then performs its Bellman update on this retrieved batch.

The bio-inspired component is **functionally integrated** into the RL loop at insertion point 3 (*experience memory / replay*) as defined in the project specification.

## 2. Architecture

```
              ┌──────────────────────────────┐
              │          Environment          │
              │        (CartPole-v1)          │
              └──────────┬───────────────────┘
                         │ (s, a, r, s', done)
                         ▼
                 ┌───────────────┐
     store ──►  │    Hopfield    │
                │   Replay Buffer │  ◄── query(s_t)
                └───────┬────────┘
                         │  mini-batch of k
                         │  retrieved transitions
                         ▼
                 ┌────────────────┐
                 │  DQN update    │
                 │  (Bellman)     │
                 └───────┬────────┘
                         │  updates
                         ▼
                 ┌────────────────┐
                 │   Q-network    │
                 └────────────────┘
```

A programmatically-generated version of this diagram is produced by
`scripts/plot_architecture.py` and included in the final report.

## 3. Repository layout

```
apbio_project/
├── notebooks/       Jupyter / Colab notebooks for development and analysis
├── src/             Core Python modules (Hopfield, replay buffer, training)
├── configs/         YAML hyperparameter configurations
├── scripts/         Experiment runners and plotting utilities
├── logs/            Raw per-seed CSV training logs
├── figures/         Figures used in the report
└── report/          LaTeX source of the scientific report
```

## 4. Installation

### Option A — Google Colab (recommended)

Open any notebook in `notebooks/` directly in Colab:

> Click the "Open in Colab" badge at the top of each notebook.

Colab already provides Python, PyTorch, and most dependencies. The first cell of every notebook installs the remaining packages automatically.

### Option B — Local installation

Requires Python ≥ 3.10.

```bash
git clone https://github.com/tosuun/apbio_project.git
cd apbio_project
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 5. Reproducing the main experiment

To reproduce the headline result (hybrid vs. baseline, 5 seeds):

```bash
python -m src.train --config configs/default.yaml --mode baseline
python -m src.train --config configs/default.yaml --mode hybrid
python scripts/plot_results.py
```

**Expected runtime:** approximately 90 minutes on a standard laptop CPU
(no GPU required). All logs are written to `logs/` and all figures to
`figures/`.

## 6. Deliverables (course requirement)

| # | Deliverable | Location |
|---|-------------|----------|
| 1 | Written report (LaTeX, ≥ 10 pages) | `report/main.pdf` |
| 2 | Self-contained runnable code | this repository |
| 3 | Experimental logs + reproduction script | `logs/`, `scripts/plot_results.py` |
| 4 | Video presentation (≈ 10 min) | *URL to be added before submission* |

## 7. License

MIT License. See `LICENSE` for details.

## 8. Academic integrity

This is an individual project. All code, experimental results, and writing
are the author's own work. External libraries and any adapted code are
cited in the report.
