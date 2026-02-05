"""
Publication-ready plots for LLM auction intervention experiments.
Metric: bid - value (signed deviation, shows over/under-bidding)

Each intervention plot shows:
- Rows: One per model
- Columns: One per intervention (excluding axis baselines - SPSB is the baseline)
- Gray histogram: SPSB baseline distribution
- Colored histogram: Intervention distribution
- Vertical lines at means
"""

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

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_NAMES = {
    'claude-3-5-haiku-20241022': 'Claude 3.5 Haiku',
    'gemini-2.0-flash': 'Gemini 2.0 Flash',
    'google/gemma-3-27b-it': 'Gemma 3 27B',
    'gpt-4o': 'GPT-4o'
}

# Clean, professional palette
MODEL_COLORS = {
    'Claude 3.5 Haiku': '#4363d8',    # Blue
    'Gemini 2.0 Flash': '#e6194B',    # Red
    'Gemma 3 27B': '#3cb44b',         # Green
    'GPT-4o': '#f58231'               # Orange
}

MODEL_ORDER = ['Claude 3.5 Haiku', 'Gemini 2.0 Flash', 'Gemma 3 27B', 'GPT-4o']

SPSB_COLOR = '#888888'  # Gray for SPSB baseline

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load the combined experimental results (most recent file)."""
    results_dir = Path(__file__).parent.parent / 'results'
    # Find the most recent combined results file
    combined_files = sorted(results_dir.glob('all_experiments_combined_*.csv'))
    if not combined_files:
        raise FileNotFoundError("No combined results files found")
    data_path = combined_files[-1]  # Most recent
    print(f"Using: {data_path.name}")
    df = pd.read_csv(data_path)
    df['model_short'] = df['model'].map(MODEL_NAMES)
    df['deviation'] = df['bid'] - df['player_value']  # Signed deviation
    return df


def get_reference_data(df):
    """Get SPSB and AC data for reference distributions."""
    refs = {}

    spsb_data = df[df['experiment'].isin(['spsb_apv', 'spsb'])]
    if not spsb_data.empty:
        refs['spsb'] = spsb_data
        refs['spsb_by_model'] = {
            model: spsb_data[spsb_data['model_short'] == model]['deviation'].values
            for model in MODEL_ORDER if model in spsb_data['model_short'].values
        }

    ac_data = df[df['experiment'].isin(['ascending_clock_apv', 'ascending_clock_closed'])]
    if not ac_data.empty:
        refs['ac'] = ac_data
        refs['ac_by_model'] = {
            model: ac_data[ac_data['model_short'] == model]['deviation'].values
            for model in MODEL_ORDER if model in ac_data['model_short'].values
        }

    return refs


# ============================================================================
# HISTOGRAM DISTRIBUTION PLOTS
# ============================================================================

def plot_intervention_histograms(df, experiments, title, filename, xlabel_map, refs):
    """
    Create a grid of histograms showing bid - value distributions.

    Layout:
    - Rows: Models
    - Columns: Interventions
    - Each cell: SPSB (gray) vs Intervention (colored)
    """
    models = [m for m in MODEL_ORDER if m in df['model_short'].unique()]
    exps = [e for e in experiments if e in df['experiment'].unique()]

    n_models = len(models)
    n_exps = len(exps)

    fig, axes = plt.subplots(n_models, n_exps, figsize=(3.0 * n_exps, 2.6 * n_models),
                             squeeze=False)

    # Determine common x-axis range (symmetric around 0)
    all_devs = df[df['experiment'].isin(experiments)]['deviation']
    if 'spsb' in refs:
        all_devs = pd.concat([all_devs, refs['spsb']['deviation']])

    x_limit = min(np.percentile(np.abs(all_devs.dropna()), 98), 25)
    bins = np.linspace(-x_limit, x_limit, 30)

    for row_idx, model in enumerate(models):
        for col_idx, exp in enumerate(exps):
            ax = axes[row_idx, col_idx]

            # Get intervention data
            int_data = df[(df['experiment'] == exp) &
                         (df['model_short'] == model)]['deviation'].values

            # Get SPSB baseline for this model
            spsb_data = refs.get('spsb_by_model', {}).get(model, np.array([]))

            # Plot SPSB baseline (gray, behind, with black outline)
            # Use weights to convert to percentage
            if len(spsb_data) > 0:
                spsb_weights = np.ones_like(spsb_data) * 100 / len(spsb_data)
                ax.hist(spsb_data, bins=bins, alpha=0.35, color=SPSB_COLOR,
                       edgecolor='#333333', linewidth=0.6, weights=spsb_weights)

                # Add median line for SPSB
                spsb_median = np.median(spsb_data)
                ax.axvline(spsb_median, color=SPSB_COLOR, linestyle='--',
                          linewidth=1.5, alpha=0.8)

            # Plot intervention (colored, on top)
            if len(int_data) > 0:
                int_weights = np.ones_like(int_data) * 100 / len(int_data)
                ax.hist(int_data, bins=bins, alpha=0.7, color=MODEL_COLORS[model],
                       edgecolor='white', linewidth=0.5, weights=int_weights)

                # Add median line for intervention (dashed)
                int_median = np.median(int_data)
                ax.axvline(int_median, color=MODEL_COLORS[model], linestyle='--',
                          linewidth=2, alpha=0.9)

            # Styling
            ax.set_xlim(-x_limit, x_limit)
            ax.set_ylim(bottom=0)

            # Column titles (intervention names) - only on top row
            if row_idx == 0:
                ax.set_title(xlabel_map.get(exp, exp).replace('\n', ' '),
                           fontweight='bold', fontsize=11, pad=8)

            # Row labels (model names) - only on left column
            if col_idx == 0:
                ax.set_ylabel(f'{model}\n% of obs', fontweight='bold', fontsize=10,
                            color=MODEL_COLORS[model])
            else:
                ax.set_ylabel('')

            # Show y-tick labels (percentage)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))

            # X-axis label only on bottom row
            if row_idx == n_models - 1:
                ax.set_xlabel('bid − value', fontsize=9)


            # Add stats annotation (median deviation) or "N/A" if missing
            if len(int_data) > 0:
                int_median = np.median(int_data)

                # Color based on direction
                if int_median > 0:
                    text_color = '#c44e52'  # Red for overbidding
                else:
                    text_color = '#2d8a2d'  # Green for underbidding

                ax.text(0.97, 0.95, f'med={int_median:+.1f}',
                       transform=ax.transAxes, fontsize=9, fontweight='bold',
                       ha='right', va='top', color=text_color,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                alpha=0.85, edgecolor='none'))
            else:
                # No data for this model/experiment
                ax.text(0.5, 0.5, 'No data',
                       transform=ax.transAxes, fontsize=10,
                       ha='center', va='center', color='#999999',
                       style='italic')

    # Add legend at bottom
    legend_elements = [
        mpatches.Patch(facecolor=SPSB_COLOR, alpha=0.35, edgecolor='#333333', linewidth=0.6, label='SPSB Baseline'),
        mpatches.Patch(facecolor='#666666', alpha=0.7, label='Intervention'),
        plt.Line2D([0], [0], color='#666666', linestyle='--', linewidth=2, label='Intervention Median'),
        plt.Line2D([0], [0], color=SPSB_COLOR, linestyle='--', linewidth=1.5, label='SPSB Median'),
    ]

    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
              frameon=True, framealpha=0.95, edgecolor='#cccccc',
              bbox_to_anchor=(0.5, -0.02), fontsize=9)

    # Main title
    fig.suptitle(title, fontweight='bold', fontsize=14, y=1.02)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {filename}")


def print_findings(df, refs):
    """Print key findings."""
    print("\n" + "═"*65)
    print("KEY FINDINGS")
    print("═"*65)

    if 'spsb' in refs:
        spsb_mean = refs['spsb']['deviation'].mean()
        print(f"\nSPSB Baseline mean(bid - value): {spsb_mean:+.2f}")

    groups = {
        'Axis 1 (Contingent)': ['axis1_contingent_dominated',
                                'axis1_contingent_enumerate', 'axis1_contingent_worstcase'],
        'Axis 2 (Forward)': ['axis2_forward_onestep',
                             'axis2_forward_tree', 'axis2_forward_backward_induct'],
        'Axis 3 (Beliefs)': ['axis3_beliefs_firstorder',
                             'axis3_beliefs_secondorder', 'axis3_beliefs_common_knowledge'],
        'Loss Aversion': ['loss_aversion_gain_frame',
                          'loss_aversion_loss_frame', 'loss_aversion_mixed_frame',
                          'loss_aversion_endowment', 'loss_aversion_WTA_WTP'],
        'Risk Preferences': ['risk_averse', 'risk_neutrality', 'risk_seeking'],
    }

    for group_name, exps in groups.items():
        print(f"\n{group_name}:")
        subset = df[df['experiment'].isin(exps)]

        for exp in exps:
            exp_data = subset[subset['experiment'] == exp]['deviation']
            if len(exp_data) > 0:
                exp_name = exp.split('_')[-1]
                print(f"  {exp_name}: mean = {exp_data.mean():+.2f}")


def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df):,} observations\n")

    refs = get_reference_data(df)

    if 'spsb' in refs:
        print(f"SPSB Baseline mean(bid - value): {refs['spsb']['deviation'].mean():+.2f}")

    # Check for missing data
    print("\nData availability:")
    for model in MODEL_ORDER:
        model_exps = df[df['model_short'] == model]['experiment'].unique()
        print(f"  {model}: {len(model_exps)} experiments")

    print("\nGenerating plots...\n")

    # ─────────────────────────────────────────────────────────────────────────
    # AXIS 1: Contingent Reasoning (no baseline - SPSB is the baseline)
    # ─────────────────────────────────────────────────────────────────────────
    axis1_exps = ['axis1_contingent_dominated',
                  'axis1_contingent_enumerate', 'axis1_contingent_worstcase']
    axis1_labels = {
        'axis1_contingent_dominated': 'Dominated',
        'axis1_contingent_enumerate': 'Enumerate',
        'axis1_contingent_worstcase': 'Worst-case'
    }
    plot_intervention_histograms(df, axis1_exps, 'Axis 1: Contingent Reasoning',
                                 'axis1_contingent_reasoning.png', axis1_labels, refs)

    # ─────────────────────────────────────────────────────────────────────────
    # AXIS 2: Forward Reasoning
    # ─────────────────────────────────────────────────────────────────────────
    axis2_exps = ['axis2_forward_onestep',
                  'axis2_forward_tree', 'axis2_forward_backward_induct']
    axis2_labels = {
        'axis2_forward_onestep': 'One-step',
        'axis2_forward_tree': 'Game Tree',
        'axis2_forward_backward_induct': 'Backward Ind.'
    }
    plot_intervention_histograms(df, axis2_exps, 'Axis 2: Forward Reasoning',
                                 'axis2_forward_reasoning.png', axis2_labels, refs)

    # ─────────────────────────────────────────────────────────────────────────
    # AXIS 3: Beliefs
    # ─────────────────────────────────────────────────────────────────────────
    axis3_exps = ['axis3_beliefs_firstorder',
                  'axis3_beliefs_secondorder', 'axis3_beliefs_common_knowledge']
    axis3_labels = {
        'axis3_beliefs_firstorder': 'First-order',
        'axis3_beliefs_secondorder': 'Second-order',
        'axis3_beliefs_common_knowledge': 'Common Know.'
    }
    plot_intervention_histograms(df, axis3_exps, 'Axis 3: Belief Reasoning',
                                 'axis3_beliefs.png', axis3_labels, refs)

    # ─────────────────────────────────────────────────────────────────────────
    # Loss Aversion (no baseline)
    # ─────────────────────────────────────────────────────────────────────────
    loss_exps = ['loss_aversion_gain_frame',
                 'loss_aversion_loss_frame', 'loss_aversion_mixed_frame',
                 'loss_aversion_endowment', 'loss_aversion_WTA_WTP']
    loss_labels = {
        'loss_aversion_gain_frame': 'Gain',
        'loss_aversion_loss_frame': 'Loss',
        'loss_aversion_mixed_frame': 'Mixed',
        'loss_aversion_endowment': 'Endowment',
        'loss_aversion_WTA_WTP': 'WTA/WTP'
    }
    plot_intervention_histograms(df, loss_exps, 'Loss Aversion Interventions',
                                 'loss_aversion.png', loss_labels, refs)

    # ─────────────────────────────────────────────────────────────────────────
    # Risk Preferences
    # ─────────────────────────────────────────────────────────────────────────
    risk_exps = ['risk_averse', 'risk_neutrality', 'risk_seeking']
    risk_labels = {
        'risk_averse': 'Risk Averse',
        'risk_neutrality': 'Risk Neutral',
        'risk_seeking': 'Risk Seeking'
    }
    plot_intervention_histograms(df, risk_exps, 'Risk Preference Interventions',
                                 'risk_preferences.png', risk_labels, refs)

    print_findings(df, refs)

    print("\n" + "═"*65)
    print(f"All plots saved to: {OUTPUT_DIR}")
    print("═"*65)


if __name__ == '__main__':
    main()
