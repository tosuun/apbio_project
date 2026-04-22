"""
Generates the architecture diagram for the report.
Run from the project root:
    python scripts/plot_architecture.py
The figure is saved to figures/architecture.png.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(ax, x, y, w, h, text, fc='white', ec='black', fontsize=10):
    bbox = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle='round,pad=0.02,rounding_size=0.02',
        linewidth=1.2, facecolor=fc, edgecolor=ec,
    )
    ax.add_patch(bbox)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, wrap=True)


def arrow(ax, x1, y1, x2, y2, label=None, style='-', color='black',
          rad=0.0, label_offset=(0, 0.12), label_color=None):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->', mutation_scale=14,
        color=color, linewidth=1.3,
        linestyle=style,
        connectionstyle=f'arc3,rad={rad}',
    )
    ax.add_patch(arr)
    if label:
        mx = (x1 + x2) / 2 + label_offset[0]
        my = (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, ha='center', fontsize=8,
                style='italic', color=label_color if label_color else color)


def main():
    out_dir = Path('figures')
    out_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # --- Boxes ---------------------------------------------------------
    box(ax, 1.5, 3.0, 2.0, 1.3, 'Environment\n(CartPole-v1)', fc='#e3eefc')
    box(ax, 6.0, 5.0, 2.8, 1.2, 'Hopfield memory\n(attention over stored $s$)', fc='#ffe9d1')
    box(ax, 6.0, 1.0, 2.8, 1.2, 'Replay buffer\n(transition arrays)', fc='#d8f3dc')
    box(ax, 10.5, 3.0, 2.0, 1.3, 'DQN\nQ-network', fc='#e3eefc')

    # --- Arrows --------------------------------------------------------
    # Env -> Hopfield (store observation)
    arrow(ax, 2.5, 3.5, 4.6, 4.7,
          label='store $s_t$')

    # Env -> Replay buffer (store full transition)
    arrow(ax, 2.5, 2.5, 4.6, 1.3,
          label="store $(s,a,r,s')$", label_offset=(0, -0.25))

    # Hopfield retrieval -> Replay buffer (gives sampling weights)
    arrow(ax, 6.0, 4.4, 6.0, 1.6,
          label=r'$w=\mathrm{softmax}(\beta X q)$',
          style='--', color='#d35400',
          label_offset=(1.7, 0))

    # Replay buffer -> DQN (mini-batch sampled according to w)
    arrow(ax, 7.4, 1.3, 9.5, 2.5,
          label='mini-batch', label_offset=(0, -0.2))

    # DQN -> Env (action) — routed BELOW the whole stack to avoid clutter
    # Using a curved arrow that goes down and left
    arrow(ax, 9.5, 3.5, 2.5, 3.5,
          label='action $a_t$', rad=0.45,
          label_offset=(0, -1.6))

    # Annotation near Hopfield clarifying the query
    ax.text(7.6, 4.95, r'query $q=s_t$ (most recent obs.)',
            fontsize=8, style='italic', color='#666666')

    # Caption note about what the dashed arrow means
    ax.text(6.0, 3.0, '(weights drive index\nsampling from buffer)',
            ha='center', fontsize=7.5, style='italic', color='#d35400')

    # --- Title --------------------------------------------------------
    ax.set_title(
        'Hybrid architecture: Hopfield replay buffer + DQN on CartPole-v1',
        fontsize=11, pad=12,
    )

    solid = mpatches.Patch(color='black', label='DQN data flow')
    dashed = mpatches.Patch(color='#d35400', label='Hopfield retrieval (new)')
    ax.legend(handles=[solid, dashed], loc='lower center',
              bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, fontsize=9)

    fig.tight_layout()
    out_path = out_dir / 'architecture.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'saved: {out_path}')


if __name__ == '__main__':
    main()
