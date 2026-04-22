"""
Regenerates all figures in the report from the raw CSVs in logs/.

Run from the project root:
    python scripts/plot_results.py

Expected inputs (as produced by the notebooks):
    logs/baseline_main_seed{0..4}.csv
    logs/hybrid_main_seed{0..4}.csv
    logs/best_seed{0..4}.csv
    logs/sweeps/beta{1.0,5.0,10.0}_seed{0,1,2}.csv
    logs/sweeps/cap{1000,10000,100000}_seed{0,1,2}.csv
    logs/sweeps/lr{0.0005,0.0023,0.005}_seed{0,1,2}.csv

Outputs go in figures/. This script replicates the plotting done in
notebook 04; it does not rerun any training.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / 'logs'
SWEEPS = LOGS / 'sweeps'
FIGS = ROOT / 'figures'
FIGS.mkdir(exist_ok=True)


# ---------- helpers ----------------------------------------------------
def load_curves(prefix, seeds, subdir=None):
    base = SWEEPS if subdir == 'sweeps' else LOGS
    out = {}
    for s in seeds:
        p = base / f'{prefix}_seed{s}.csv'
        df = pd.read_csv(p)
        df['smooth'] = df['return'].rolling(20, min_periods=1).mean()
        out[s] = df
    return out


def stack(curves, n_points=200):
    t_max = min(df['timestep'].max() for df in curves.values())
    grid = np.linspace(0, t_max, n_points)
    rows = [np.interp(grid, df['timestep'].values, df['smooth'].values)
            for df in curves.values()]
    stacked = np.vstack(rows)
    return grid, stacked.mean(axis=0), stacked.std(axis=0)


def final_returns(curves):
    return [df['return'].tail(max(10, int(0.1 * len(df)))).mean()
            for df in curves.values()]


# ---------- individual plots ------------------------------------------
def plot_main(seeds):
    base = load_curves('baseline_main', seeds)
    hyb = load_curves('hybrid_main', seeds)
    gb, mb, sb = stack(base)
    gh, mh, sh = stack(hyb)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(gb, mb, label='baseline (mean over 5 seeds)', color='tab:blue')
    ax.fill_between(gb, mb - sb, mb + sb, color='tab:blue', alpha=0.2)
    ax.plot(gh, mh, label='hybrid (mean over 5 seeds)', color='tab:orange')
    ax.fill_between(gh, mh - sh, mh + sh, color='tab:orange', alpha=0.2)
    ax.axhline(500, color='grey', linestyle='--', linewidth=1, label='max (500)')
    ax.set_xlabel('Environment timestep')
    ax.set_ylabel('Return (20-ep rolling mean)')
    ax.set_title('Main experiment \u2014 hybrid vs. baseline DQN on CartPole-v1 (5 seeds)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / 'main_hybrid_vs_baseline.png', dpi=150)
    plt.close(fig)
    print('wrote main_hybrid_vs_baseline.png')

    # Significance test
    fr_b = final_returns(base)
    fr_h = final_returns(hyb)
    t, p = stats.ttest_ind(fr_h, fr_b, equal_var=False)
    rel = (np.mean(fr_h) - np.mean(fr_b)) / abs(np.mean(fr_b))
    print(f'  baseline final = {np.mean(fr_b):.1f} \u00b1 {np.std(fr_b):.1f}')
    print(f'  hybrid   final = {np.mean(fr_h):.1f} \u00b1 {np.std(fr_h):.1f}')
    print(f'  rel. improvement = {rel*100:+.1f}%')
    print(f'  Welch t-test: t={t:.2f}, p={p:.3f}')


def plot_sweep(prefix, values, seeds, title, xlabel_fmt, out_name):
    fig, ax = plt.subplots(figsize=(9, 5))
    for v in values:
        key = f'{prefix}{v}'
        curves = load_curves(key, seeds, subdir='sweeps')
        g, m, sd = stack(curves)
        ax.plot(g, m, label=xlabel_fmt.format(v))
        ax.fill_between(g, m - sd, m + sd, alpha=0.2)
    ax.axhline(500, color='grey', linestyle='--', linewidth=1)
    ax.set_xlabel('Environment timestep')
    ax.set_ylabel('Return (20-ep rolling mean)')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / out_name, dpi=150)
    plt.close(fig)
    print(f'wrote {out_name}')


def plot_best(seeds):
    base = load_curves('baseline_main', seeds)
    best = load_curves('best', seeds)
    gb, mb, sb = stack(base)
    gh, mh, sh = stack(best)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(gb, mb, label='baseline DQN', color='tab:blue')
    ax.fill_between(gb, mb - sb, mb + sb, color='tab:blue', alpha=0.2)
    ax.plot(gh, mh, label='best hybrid ($\\beta$=1.0)', color='tab:green')
    ax.fill_between(gh, mh - sh, mh + sh, color='tab:green', alpha=0.2)
    ax.axhline(500, color='grey', linestyle='--', linewidth=1, label='max (500)')
    ax.set_xlabel('Environment timestep')
    ax.set_ylabel('Return (20-ep rolling mean)')
    ax.set_title('Best hybrid configuration vs. baseline DQN (5 seeds)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / 'best_vs_baseline.png', dpi=150)
    plt.close(fig)
    print('wrote best_vs_baseline.png')

    fr_b = final_returns(base)
    fr_h = final_returns(best)
    t, p = stats.ttest_ind(fr_h, fr_b, equal_var=False)
    rel = (np.mean(fr_h) - np.mean(fr_b)) / abs(np.mean(fr_b))
    print(f'  baseline final = {np.mean(fr_b):.1f} \u00b1 {np.std(fr_b):.1f}')
    print(f'  best     final = {np.mean(fr_h):.1f} \u00b1 {np.std(fr_h):.1f}')
    print(f'  rel. improvement = {rel*100:+.1f}%')
    print(f'  Welch t-test: t={t:.2f}, p={p:.3f}')


# ---------- main ------------------------------------------------------
def main():
    MAIN_SEEDS = [0, 1, 2, 3, 4]
    SWEEP_SEEDS = [0, 1, 2]

    print('--- main experiment ---')
    plot_main(MAIN_SEEDS)

    print('\n--- sweep 1: beta ---')
    plot_sweep('beta', [1.0, 5.0, 10.0], SWEEP_SEEDS,
               'Sweep 1 \u2014 Hopfield $\\beta$ (3 seeds each)',
               '$\\beta$ = {}', 'sweep_beta.png')

    print('\n--- sweep 2: capacity ---')
    plot_sweep('cap', [1000, 10000, 100000], SWEEP_SEEDS,
               'Sweep 2 \u2014 Hopfield capacity N (3 seeds each)',
               'N = {}', 'sweep_capacity.png')

    print('\n--- sweep 3: learning rate ---')
    plot_sweep('lr', [0.0005, 0.0023, 0.005], SWEEP_SEEDS,
               'Sweep 3 \u2014 DQN learning rate (3 seeds each)',
               'lr = {}', 'sweep_lr.png')

    print('\n--- best config ---')
    plot_best(MAIN_SEEDS)

    print('\nall figures written to figures/')


if __name__ == '__main__':
    main()
