#!/usr/bin/env python3
"""
Analyze and visualize LoRA scan sweep results.

Loads sft_scan_results.csv and produces:
1. Summary statistics (mean/std across seeds)
2. Performance comparisons across models, datasets, and prompts
3. Publication-quality visualizations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Configuration
INPUT_CSV = "sft_scan_results.csv"
OUTPUT_DIR = Path("analysis_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Visualization settings (colorblind-friendly palette)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

# Colorblind-safe categorical palette (based on dataviz skill guidelines)
# Using a fixed hue order that works for CVD
COLORS = {
    'qwen25-0.5b': '#0066CC',   # Blue
    'qwen25-1.5b': '#109618',   # Green
    'qwen25-3b': '#FF9900',     # Orange
    'gemma2-2b': '#990099',     # Purple
}

PROMPT_COLORS = {
    'answer_only': '#0066CC',   # Blue
    'chat': '#109618',          # Green
    'cot': '#FF9900',           # Orange
}

DATASET_MARKERS = {
    'mmlu': 'o',
    'gsm8k': 's',
}


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and validate the results CSV."""
    df = pd.read_csv(csv_path, index_col=0)
    print(f"Loaded {len(df)} runs")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    print(f"Prompts: {sorted(df['prompt'].unique())}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    return df


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results across seeds: mean ± std for each configuration."""
    # Group by all experimental factors except seed
    group_cols = ['model', 'dataset', 'prompt', 'rank', 'location']

    agg_dict = {
        'accuracy': ['mean', 'std', 'count'],
        'train_final_train_loss': ['mean', 'std'],
        'train_trainable_params': 'first',
        'mean_completion_tokens': ['mean', 'std'],
        'n_unparsed': 'mean',
    }

    summary = df.groupby(group_cols).agg(agg_dict).reset_index()

    # Flatten column names
    summary.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                       for col in summary.columns.values]

    # Rename for clarity
    summary = summary.rename(columns={
        'accuracy_mean': 'accuracy',
        'accuracy_std': 'accuracy_std',
        'accuracy_count': 'n_seeds',
        'train_final_train_loss_mean': 'train_loss',
        'train_final_train_loss_std': 'train_loss_std',
        'train_trainable_params_first': 'train_trainable_params',
        'mean_completion_tokens_mean': 'mean_tokens',
        'mean_completion_tokens_std': 'mean_tokens_std',
    })

    return summary


def print_summary_table(summary: pd.DataFrame):
    """Print formatted summary table."""
    print("\n" + "="*80)
    print("SUMMARY: Accuracy by Model × Dataset × Prompt (mean ± std across 3 seeds)")
    print("="*80)

    for dataset in sorted(summary['dataset'].unique()):
        print(f"\n{dataset.upper()}")
        print("-" * 70)
        subset = summary[summary['dataset'] == dataset].copy()
        subset = subset.sort_values(['model', 'prompt'])

        for _, row in subset.iterrows():
            acc = row['accuracy'] * 100
            std = row['accuracy_std'] * 100
            print(f"  {row['model']:15s} | {row['prompt']:12s} | "
                  f"{acc:5.2f} ± {std:4.2f}%  "
                  f"(params={row['train_trainable_params']:,})")


def plot_accuracy_by_model_dataset(df: pd.DataFrame, summary: pd.DataFrame):
    """Bar chart: accuracy by model for each dataset×prompt combination."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    fig.suptitle('Accuracy by Model, Dataset, and Prompt Strategy', fontsize=14, y=0.995)

    datasets = sorted(df['dataset'].unique())
    prompts = sorted(df['prompt'].unique())
    models = sorted(df['model'].unique())

    for i, dataset in enumerate(datasets):
        for j, prompt in enumerate(prompts):
            ax = axes[i, j]

            subset = summary[(summary['dataset'] == dataset) &
                           (summary['prompt'] == prompt)].copy()
            subset = subset.sort_values('model')

            x = np.arange(len(subset))
            bars = ax.bar(x, subset['accuracy'] * 100,
                         yerr=subset['accuracy_std'] * 100,
                         color=[COLORS[m] for m in subset['model']],
                         capsize=4, alpha=0.9, width=0.7)

            ax.set_xticks(x)
            ax.set_xticklabels([m.replace('qwen25-', 'Q') for m in subset['model']],
                              rotation=45, ha='right')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(f"{dataset} / {prompt}")
            ax.grid(axis='y', alpha=0.3, linewidth=0.5)
            ax.set_ylim(0, 100)

            # Add value labels on bars
            for bar, val in zip(bars, subset['accuracy'] * 100):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'accuracy_by_model_dataset_prompt.png', bbox_inches='tight')
    print(f"\n✓ Saved: {OUTPUT_DIR / 'accuracy_by_model_dataset_prompt.png'}")


def plot_heatmap_accuracy(summary: pd.DataFrame):
    """Heatmap of accuracy for each dataset."""
    datasets = sorted(summary['dataset'].unique())

    fig, axes = plt.subplots(1, len(datasets), figsize=(14, 5))
    if len(datasets) == 1:
        axes = [axes]

    fig.suptitle('Accuracy Heatmap: Model × Prompt', fontsize=14)

    for ax, dataset in zip(axes, datasets):
        subset = summary[summary['dataset'] == dataset].copy()

        # Pivot for heatmap
        pivot = subset.pivot_table(values='accuracy',
                                   index='prompt',
                                   columns='model')
        pivot = pivot * 100  # Convert to percentage

        # Reorder rows and columns
        pivot = pivot.reindex(sorted(pivot.index))
        pivot = pivot[sorted(pivot.columns)]

        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn',
                   vmin=0, vmax=100, cbar_kws={'label': 'Accuracy (%)'},
                   ax=ax, linewidths=1, linecolor='white')
        ax.set_title(f'{dataset.upper()}')
        ax.set_xlabel('Model')
        ax.set_ylabel('Prompt Strategy')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'accuracy_heatmap.png', bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'accuracy_heatmap.png'}")


def plot_tokens_by_model_dataset(df: pd.DataFrame, summary: pd.DataFrame):
    """Bar chart: mean completion tokens by model for each dataset×prompt combination."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    fig.suptitle('Mean Completion Tokens by Model, Dataset, and Prompt Strategy', fontsize=14, y=0.995)

    datasets = sorted(df['dataset'].unique())
    prompts = sorted(df['prompt'].unique())
    models = sorted(df['model'].unique())

    for i, dataset in enumerate(datasets):
        for j, prompt in enumerate(prompts):
            ax = axes[i, j]

            subset = summary[(summary['dataset'] == dataset) &
                           (summary['prompt'] == prompt)].copy()
            subset = subset.sort_values('model')

            x = np.arange(len(subset))
            bars = ax.bar(x, subset['mean_tokens'],
                         yerr=subset['mean_tokens_std'],
                         color=[COLORS[m] for m in subset['model']],
                         capsize=4, alpha=0.9, width=0.7)

            ax.set_xticks(x)
            ax.set_xticklabels([m.replace('qwen25-', 'Q') for m in subset['model']],
                              rotation=45, ha='right')
            ax.set_ylabel('Mean Completion Tokens')
            ax.set_title(f"{dataset} / {prompt}")
            ax.grid(axis='y', alpha=0.3, linewidth=0.5)

            # Add value labels on bars
            for bar, val in zip(bars, subset['mean_tokens']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{val:.0f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'tokens_by_model_dataset_prompt.png', bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'tokens_by_model_dataset_prompt.png'}")


def plot_prompt_comparison(summary: pd.DataFrame):
    """Compare prompt strategies within each model and dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Prompt Strategy Comparison by Model', fontsize=14)

    datasets = sorted(summary['dataset'].unique())
    models = sorted(summary['model'].unique())
    prompts = sorted(summary['prompt'].unique())

    for ax, dataset in zip(axes, datasets):
        # Group bars by model
        x = np.arange(len(models))
        width = 0.27

        for i, prompt in enumerate(prompts):
            subset = summary[(summary['dataset'] == dataset) &
                           (summary['prompt'] == prompt)].copy()
            subset = subset.sort_values('model')

            offset = width * (i - 1)
            ax.bar(x + offset, subset['accuracy'] * 100,
                  width, label=prompt, color=PROMPT_COLORS[prompt],
                  alpha=0.9, yerr=subset['accuracy_std'] * 100, capsize=3)

        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{dataset.upper()}')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(title='Prompt', loc='best')
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'prompt_strategy_comparison.png', bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'prompt_strategy_comparison.png'}")


def plot_tokens_prompt_comparison(summary: pd.DataFrame):
    """Compare prompt strategies by tokens within each model and dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Mean Completion Tokens: Prompt Strategy Comparison by Model', fontsize=14)

    datasets = sorted(summary['dataset'].unique())
    models = sorted(summary['model'].unique())
    prompts = sorted(summary['prompt'].unique())

    for ax, dataset in zip(axes, datasets):
        # Group bars by model
        x = np.arange(len(models))
        width = 0.27

        for i, prompt in enumerate(prompts):
            subset = summary[(summary['dataset'] == dataset) &
                           (summary['prompt'] == prompt)].copy()
            subset = subset.sort_values('model')

            offset = width * (i - 1)
            ax.bar(x + offset, subset['mean_tokens'],
                  width, label=prompt, color=PROMPT_COLORS[prompt],
                  alpha=0.9, yerr=subset['mean_tokens_std'], capsize=3)

        ax.set_xlabel('Model')
        ax.set_ylabel('Mean Completion Tokens')
        ax.set_title(f'{dataset.upper()}')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(title='Prompt', loc='best')
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'tokens_prompt_strategy_comparison.png', bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'tokens_prompt_strategy_comparison.png'}")


def plot_training_diagnostics(df: pd.DataFrame):
    """Scatter plots of training loss vs accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Diagnostics', fontsize=14)

    datasets = sorted(df['dataset'].unique())

    for ax, dataset in zip(axes, datasets):
        subset = df[df['dataset'] == dataset]

        for model in sorted(subset['model'].unique()):
            model_data = subset[subset['model'] == model]
            ax.scatter(model_data['train_final_train_loss'],
                      model_data['accuracy'] * 100,
                      label=model, color=COLORS[model],
                      alpha=0.6, s=50, edgecolors='white', linewidth=0.5)

        ax.set_xlabel('Final Training Loss')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title(f'{dataset.upper()}')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_diagnostics.png', bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'training_diagnostics.png'}")


def plot_parameter_efficiency(summary: pd.DataFrame):
    """Accuracy vs trainable parameters."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Parameter Efficiency', fontsize=14)

    datasets = sorted(summary['dataset'].unique())

    for ax, dataset in zip(axes, datasets):
        subset = summary[summary['dataset'] == dataset]

        for model in sorted(subset['model'].unique()):
            model_data = subset[subset['model'] == model]

            # Plot with error bars
            ax.errorbar(model_data['train_trainable_params'],
                       model_data['accuracy'] * 100,
                       yerr=model_data['accuracy_std'] * 100,
                       fmt='o', label=model, color=COLORS[model],
                       markersize=8, capsize=4, alpha=0.8)

        ax.set_xlabel('Trainable Parameters')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{dataset.upper()}')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'parameter_efficiency.png', bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'parameter_efficiency.png'}")


def plot_seed_variance(df: pd.DataFrame):
    """Visualize variance across seeds for each configuration."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Variance Across Random Seeds', fontsize=14)

    datasets = sorted(df['dataset'].unique())

    for ax, dataset in zip(axes, datasets):
        subset = df[df['dataset'] == dataset].copy()
        subset['config'] = subset['model'] + ' / ' + subset['prompt']

        configs = sorted(subset['config'].unique())

        # Prepare data for box plot
        data_by_config = [subset[subset['config'] == c]['accuracy'].values * 100
                         for c in configs]

        bp = ax.boxplot(data_by_config, tick_labels=configs, patch_artist=True,
                       showmeans=True, meanline=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2),
                       meanprops=dict(color='green', linewidth=2, linestyle='--'))

        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{dataset.upper()}')
        ax.tick_params(axis='x', rotation=45)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'seed_variance.png', bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'seed_variance.png'}")


def export_summary_tables(summary: pd.DataFrame, df: pd.DataFrame):
    """Export summary statistics to CSV and JSON."""
    # Main summary table
    summary_export = summary.copy()
    summary_export['accuracy_pct'] = summary_export['accuracy'] * 100
    summary_export['accuracy_std_pct'] = summary_export['accuracy_std'] * 100
    summary_export.to_csv(OUTPUT_DIR / 'summary_statistics.csv', index=False)
    print(f"✓ Saved: {OUTPUT_DIR / 'summary_statistics.csv'}")

    # Best configurations
    best_by_dataset = []
    for dataset in sorted(summary['dataset'].unique()):
        subset = summary[summary['dataset'] == dataset]
        best_idx = subset['accuracy'].idxmax()
        best = subset.loc[best_idx].copy()
        best['dataset'] = dataset
        best_by_dataset.append(best)

    best_df = pd.DataFrame(best_by_dataset)
    best_df.to_csv(OUTPUT_DIR / 'best_configurations.csv', index=False)
    print(f"✓ Saved: {OUTPUT_DIR / 'best_configurations.csv'}")

    # Summary statistics as JSON
    stats = {
        'total_runs': len(df),
        'n_models': len(df['model'].unique()),
        'n_datasets': len(df['dataset'].unique()),
        'n_prompts': len(df['prompt'].unique()),
        'n_seeds': len(df['seed'].unique()),
        'best_by_dataset': {
            row['dataset']: {
                'model': row['model'],
                'prompt': row['prompt'],
                'accuracy': float(row['accuracy']),
                'accuracy_std': float(row['accuracy_std']),
            }
            for _, row in best_df.iterrows()
        }
    }

    with open(OUTPUT_DIR / 'summary.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved: {OUTPUT_DIR / 'summary.json'}")


def main():
    """Run full analysis pipeline."""
    print("="*80)
    print("LoRA Scan Sweep Analysis")
    print("="*80)

    # Load data
    df = load_data(INPUT_CSV)

    # Compute summary statistics
    summary = compute_summary_stats(df)
    print_summary_table(summary)

    # Generate all visualizations
    print("\n" + "="*80)
    print("Generating Visualizations")
    print("="*80)

    plot_accuracy_by_model_dataset(df, summary)
    plot_heatmap_accuracy(summary)
    plot_prompt_comparison(summary)
    plot_tokens_by_model_dataset(df, summary)
    plot_tokens_prompt_comparison(summary)
    plot_training_diagnostics(df)
    plot_parameter_efficiency(summary)
    plot_seed_variance(df)

    # Export tables
    print("\n" + "="*80)
    print("Exporting Summary Tables")
    print("="*80)
    export_summary_tables(summary, df)

    print("\n" + "="*80)
    print("✓ Analysis Complete!")
    print(f"All outputs saved to: {OUTPUT_DIR.absolute()}")
    print("="*80)


if __name__ == "__main__":
    main()
