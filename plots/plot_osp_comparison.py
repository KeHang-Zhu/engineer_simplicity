"""
OSP/Iterative Mechanisms Comparison Plot

Creates a side-by-side figure showing:
- Left: Auctions - SPSB (gray) vs Ascending Clock (colored)
- Right: DA - Direct baseline (gray) vs OSP (colored)

Shows that OSP/iterative mechanisms improve play in both domains.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ============================================================================
# STYLE CONFIGURATION
# ============================================================================

plt.style.use('seaborn-v0_8-whitegrid')

plt.rcParams.update({
    'figure.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,

    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,

    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'axes.edgecolor': '#555555',
    'axes.labelcolor': '#333333',

    'grid.alpha': 0.4,
    'grid.linewidth': 0.5,
    'axes.grid': True,
    'axes.axisbelow': True,
})

OUTPUT_DIR = Path(__file__).parent

# Colors
BASELINE_COLOR = '#888888'      # Gray for baselines
AUCTION_COLOR = '#4363d8'       # Blue for ascending clock
DA_COLOR = '#e6194B'            # Red for OSP

# ============================================================================
# DATA LOADING - AUCTIONS
# ============================================================================

MODEL_NAMES = {
    'claude-3-5-haiku-20241022': 'Claude 3.5 Haiku',
    'gemini-2.0-flash': 'Gemini 2.0 Flash',
    'google/gemma-3-27b-it': 'Gemma 3 27B',
    'gpt-4o': 'GPT-4o'
}

def load_auction_data():
    """Load the combined auction experimental results."""
    results_dir = Path(__file__).parent.parent / 'results'
    combined_files = sorted(results_dir.glob('all_experiments_combined_*.csv'))
    if not combined_files:
        raise FileNotFoundError("No combined results files found")
    data_path = combined_files[-1]
    print(f"Auctions: {data_path.name}")
    df = pd.read_csv(data_path)
    df['model_short'] = df['model'].map(MODEL_NAMES)
    df['deviation'] = df['bid'] - df['player_value']
    return df


# ============================================================================
# DATA LOADING - DA
# ============================================================================

DA_MODEL_NAMES = {
    'claude': 'Claude 3.5 Haiku',
    'gemini': 'Gemini 2.0 Flash',
    'gpt4o': 'GPT-4o',
    'gemma': 'Gemma 3 27B',
    'others': 'GPT-4o',
}

def get_true_ranking(values):
    """Convert values dict to true ranking (sorted by value, descending)."""
    return sorted(values.keys(), key=lambda x: values[x], reverse=True)


def kendall_tau_distance(true_ranking, submitted_ranking):
    """Count discordant pairs between two rankings."""
    common = set(true_ranking) & set(submitted_ranking)
    n = len(common)
    if n < 2:
        return 0, 0

    true_pos = {item: i for i, item in enumerate(true_ranking) if item in common}
    sub_pos = {item: i for i, item in enumerate(submitted_ranking) if item in common}

    items = list(common)
    discordant = 0
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            a, b = items[i], items[j]
            true_order = true_pos[a] < true_pos[b]
            sub_order = sub_pos[a] < sub_pos[b]
            if true_order != sub_order:
                discordant += 1

    n_pairs = n * (n - 1) // 2
    return discordant, n_pairs


def load_da_data():
    """Load all DA experiment results from JSON files."""
    da_dir = Path(__file__).parent.parent / 'experiment_logs' / 'da'
    rows = []

    for model_dir in da_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        if model_name not in DA_MODEL_NAMES:
            continue

        model_short = DA_MODEL_NAMES[model_name]

        for exp_dir in model_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            exp_name = exp_dir.name
            raw_data_dir = exp_dir / 'raw_data'

            if not raw_data_dir.exists():
                continue

            for json_file in raw_data_dir.glob('*.json'):
                try:
                    with open(json_file) as f:
                        data = json.load(f)

                    mechanism_type = data.get('mechanism_type', 'direct')
                    values = data.get('values', {})

                    if mechanism_type == 'osp':
                        rankings = data.get('osp_choices', {})
                    else:
                        rankings = data.get('rankings', {})

                    for student, student_values in values.items():
                        true_ranking = get_true_ranking(student_values)
                        submitted = rankings.get(student, [])

                        if not submitted:
                            continue

                        if mechanism_type == 'osp':
                            revealed_items = set(submitted)
                            true_ranking_truncated = [s for s in true_ranking if s in revealed_items]
                        else:
                            true_ranking_truncated = true_ranking

                        discordant, n_pairs = kendall_tau_distance(true_ranking_truncated, submitted)
                        normalized = discordant / n_pairs if n_pairs > 0 else 0.0

                        rows.append({
                            'model_short': model_short,
                            'experiment': exp_name,
                            'kendall_tau_normalized': normalized,
                            'mechanism_type': mechanism_type,
                        })

                except Exception as e:
                    continue

    return pd.DataFrame(rows)


# ============================================================================
# MAIN PLOT
# ============================================================================

def plot_osp_comparison():
    """
    Create side-by-side comparison showing OSP/iterative mechanisms improve play.

    Left panel: Auctions (SPSB baseline vs Ascending Clock)
    Right panel: DA (Direct baseline vs OSP)
    """
    print("Loading data...")

    # Load auction data
    auction_df = load_auction_data()
    spsb_dev = auction_df[auction_df['experiment'].isin(['spsb_apv', 'spsb'])]['deviation'].values
    ac_dev = auction_df[auction_df['experiment'].isin(['ascending_clock_apv', 'ascending_clock_closed'])]['deviation'].values

    print(f"  SPSB: n={len(spsb_dev)}, mean={np.mean(spsb_dev):+.2f}")
    print(f"  Ascending Clock: n={len(ac_dev)}, mean={np.mean(ac_dev):+.2f}")

    # Load DA data
    da_df = load_da_data()
    direct_tau = da_df[da_df['experiment'] == 'direct_baseline']['kendall_tau_normalized'].values
    osp_tau = da_df[da_df['experiment'] == 'osp_baseline']['kendall_tau_normalized'].values

    print(f"  Direct DA: n={len(direct_tau)}, mean={np.mean(direct_tau)*100:.1f}%")
    print(f"  OSP DA: n={len(osp_tau)}, mean={np.mean(osp_tau)*100:.1f}%")

    # Create figure with 2 panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # ─────────────────────────────────────────────────────────────────────────
    # LEFT PANEL: Auctions
    # ─────────────────────────────────────────────────────────────────────────
    x_limit = 20
    auction_bins = np.linspace(-x_limit, x_limit, 30)

    # Plot Ascending Clock FIRST (colored, behind)
    if len(ac_dev) > 0:
        ac_weights = np.ones_like(ac_dev) * 100 / len(ac_dev)
        ax1.hist(ac_dev, bins=auction_bins, alpha=0.6, color=AUCTION_COLOR,
                edgecolor='#2a3d8a', linewidth=0.6, weights=ac_weights,
                label='Ascending Clock')
        ac_mean = np.mean(ac_dev)
        ax1.axvline(ac_mean, color=AUCTION_COLOR, linestyle='--', linewidth=2, alpha=0.9)

    # Plot SPSB SECOND (gray, in front)
    if len(spsb_dev) > 0:
        spsb_weights = np.ones_like(spsb_dev) * 100 / len(spsb_dev)
        ax1.hist(spsb_dev, bins=auction_bins, alpha=0.5, color=BASELINE_COLOR,
                edgecolor='#333333', linewidth=0.6, weights=spsb_weights,
                label='SPSB Baseline')
        spsb_mean = np.mean(spsb_dev)
        ax1.axvline(spsb_mean, color=BASELINE_COLOR, linestyle='--', linewidth=1.5, alpha=0.8)

    # Add reference line at 0 (truthful bidding)
    ax1.axvline(0, color='#2d8a2d', linestyle='-', linewidth=1.5, alpha=0.7)

    ax1.set_xlim(-x_limit, x_limit)
    ax1.set_xlabel('bid − value', fontsize=11)
    ax1.set_ylabel('% of observations', fontsize=11)
    ax1.set_title('Auctions', fontweight='bold', fontsize=13, pad=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))

    # Annotation for auction means
    ax1.text(0.97, 0.95, f'SPSB μ={spsb_mean:+.1f}',
            transform=ax1.transAxes, fontsize=10, fontweight='bold',
            ha='right', va='top', color=BASELINE_COLOR,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none'))
    ax1.text(0.97, 0.82, f'Asc. Clock μ={ac_mean:+.1f}',
            transform=ax1.transAxes, fontsize=10, fontweight='bold',
            ha='right', va='top', color=AUCTION_COLOR,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none'))

    # ─────────────────────────────────────────────────────────────────────────
    # RIGHT PANEL: DA
    # ─────────────────────────────────────────────────────────────────────────
    da_bins = np.linspace(0, 1, 21)

    # Plot OSP FIRST (colored, behind)
    if len(osp_tau) > 0:
        osp_weights = np.ones_like(osp_tau) * 100 / len(osp_tau)
        ax2.hist(osp_tau, bins=da_bins, alpha=0.6, color=DA_COLOR,
                edgecolor='#a11232', linewidth=0.6, weights=osp_weights,
                label='OSP (Iterative)')
        osp_mean = np.mean(osp_tau)
        ax2.axvline(osp_mean, color=DA_COLOR, linestyle='--', linewidth=2, alpha=0.9)

    # Plot Direct SECOND (gray, in front)
    if len(direct_tau) > 0:
        direct_weights = np.ones_like(direct_tau) * 100 / len(direct_tau)
        ax2.hist(direct_tau, bins=da_bins, alpha=0.5, color=BASELINE_COLOR,
                edgecolor='#333333', linewidth=0.6, weights=direct_weights,
                label='Direct Baseline')
        direct_mean = np.mean(direct_tau)
        ax2.axvline(direct_mean, color=BASELINE_COLOR, linestyle='--', linewidth=1.5, alpha=0.8)

    # Add reference line at 0 (perfect play)
    ax2.axvline(0, color='#2d8a2d', linestyle='-', linewidth=1.5, alpha=0.7)

    ax2.set_xlim(0, 1)
    ax2.set_xlabel('Kendall τ (% pairs wrong)', fontsize=11)
    ax2.set_ylabel('% of observations', fontsize=11)
    ax2.set_title('Deferred Acceptance', fontweight='bold', fontsize=13, pad=10)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))

    # Annotation for DA means
    ax2.text(0.97, 0.95, f'Direct μ={direct_mean*100:.0f}%',
            transform=ax2.transAxes, fontsize=10, fontweight='bold',
            ha='right', va='top', color=BASELINE_COLOR,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none'))
    ax2.text(0.97, 0.82, f'OSP μ={osp_mean*100:.0f}%',
            transform=ax2.transAxes, fontsize=10, fontweight='bold',
            ha='right', va='top', color=DA_COLOR,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none'))

    # ─────────────────────────────────────────────────────────────────────────
    # LEGEND
    # ─────────────────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor=BASELINE_COLOR, alpha=0.5, edgecolor='#333333',
                      linewidth=0.6, label='Static Mechanism (SPSB / Direct DA)'),
        mpatches.Patch(facecolor='#666666', alpha=0.6, edgecolor='#333333',
                      linewidth=0.6, label='Iterative Mechanism (Ascending Clock / OSP)'),
        plt.Line2D([0], [0], color='#2d8a2d', linestyle='-', linewidth=1.5,
                  label='Truthful Play'),
        plt.Line2D([0], [0], color='#666666', linestyle='--', linewidth=2,
                  label='Mean'),
    ]

    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
              frameon=True, framealpha=0.95, edgecolor='#cccccc',
              bbox_to_anchor=(0.5, -0.02), fontsize=9)

    # Main title
    fig.suptitle('Iterative/OSP Mechanisms Improve Play',
                fontweight='bold', fontsize=14, y=1.02)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    output_path = OUTPUT_DIR / 'osp_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✓ Saved: {output_path}")


if __name__ == '__main__':
    plot_osp_comparison()
