#!/usr/bin/env python3
"""
Analyze and visualize experiment results from Kaggle runs.

Usage:
    python analyze_results.py                          # Analyze all experiments
    python analyze_results.py --exp exp001             # Analyze specific experiment
    python analyze_results.py --compare exp001 exp003  # Compare experiments
    python analyze_results.py --plot                   # Generate plots
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plots will be skipped")


def load_experiment(exp_dir: str) -> Dict:
    """Load experiment metrics and config."""
    exp_path = Path(exp_dir)
    
    if not exp_path.exists():
        print(f"Warning: {exp_dir} not found")
        return None
    
    result = {'exp_id': exp_path.name}
    
    # Load metrics
    metrics_file = exp_path / 'metrics.json'
    if metrics_file.exists():
        with open(metrics_file) as f:
            result['metrics'] = json.load(f)
    
    # Load config
    config_file = exp_path / 'config.json'
    if config_file.exists():
        with open(config_file) as f:
            result['config'] = json.load(f)
    
    # Load samples
    samples_file = exp_path / 'samples.txt'
    if samples_file.exists():
        with open(samples_file) as f:
            result['samples'] = f.read()
    
    return result


def analyze_experiment(exp: Dict):
    """Print analysis of a single experiment."""
    print(f"\n{'='*60}")
    print(f"Experiment: {exp['exp_id']}")
    print(f"{'='*60}")
    
    if 'config' in exp:
        config = exp['config']
        print(f"Model: {config.get('model', 'unknown')}")
        print(f"Dimensions: d_model={config.get('d_model', '?')}, d_state={config.get('d_state', '?')}")
        print(f"Latent steps: {config.get('latent_steps', 0)}")
        print(f"Epochs: {config.get('num_epochs', 0)}")
    
    if 'metrics' in exp:
        metrics = exp['metrics']
        
        train_losses = metrics.get('train_loss', [])
        val_losses = metrics.get('val_loss', [])
        
        if train_losses:
            print(f"\nTraining Loss:")
            print(f"  Initial: {train_losses[0]:.4f}")
            print(f"  Final: {train_losses[-1]:.4f}")
            print(f"  Best: {min(train_losses):.4f} (epoch {train_losses.index(min(train_losses)) + 1})")
            print(f"  Improvement: {(1 - train_losses[-1]/train_losses[0])*100:.1f}%")
        
        if val_losses:
            print(f"\nValidation Loss:")
            print(f"  Initial: {val_losses[0]:.4f}")
            print(f"  Final: {val_losses[-1]:.4f}")
            print(f"  Best: {min(val_losses):.4f} (epoch {val_losses.index(min(val_losses)) + 1})")
            print(f"  Improvement: {(1 - val_losses[-1]/val_losses[0])*100:.1f}%")
            
            # Check for overfitting
            if train_losses and val_losses:
                gap = val_losses[-1] - train_losses[-1]
                if gap > 0.1:
                    print(f"  ⚠️  Overfitting detected (gap: {gap:.4f})")
        
        # QA accuracy
        eval_accuracy = metrics.get('eval_accuracy', [])
        if eval_accuracy:
            final_eval = eval_accuracy[-1]
            print(f"\nQA Accuracy (final):")
            print(f"  Overall: {final_eval.get('overall_accuracy', 0):.3f}")
            task_acc = final_eval.get('task_accuracy', {})
            for task, acc in task_acc.items():
                print(f"  {task}: {acc:.3f}")
    
    # Sample outputs
    if 'samples' in exp:
        print(f"\nSample Generations:")
        print(exp['samples'][:500])  # First 500 chars


def compare_experiments(exp_ids: List[str]):
    """Compare multiple experiments."""
    experiments = []
    for exp_id in exp_ids:
        exp = load_experiment(f"experiments/{exp_id}")
        if exp:
            experiments.append(exp)
    
    if len(experiments) < 2:
        print("Need at least 2 experiments to compare")
        return
    
    print(f"\n{'='*80}")
    print(f"COMPARISON: {' vs '.join(exp_ids)}")
    print(f"{'='*80}")
    
    # Create comparison table
    print(f"\n{'Experiment':<12} {'Model':<20} {'Val Loss':<12} {'QA Acc':<12} {'Params':<12}")
    print('-' * 80)
    
    for exp in experiments:
        exp_id = exp['exp_id']
        config = exp.get('config', {})
        metrics = exp.get('metrics', {})
        
        model = config.get('model', 'unknown')
        val_loss = metrics.get('val_loss', [])
        final_val = f"{val_loss[-1]:.4f}" if val_loss else "N/A"
        
        eval_acc = metrics.get('eval_accuracy', [])
        qa_acc = f"{eval_acc[-1].get('overall_accuracy', 0):.3f}" if eval_acc else "N/A"
        
        # Load model to count params
        params = "?"  # Could load model.pt and count
        
        print(f"{exp_id:<12} {model:<20} {final_val:<12} {qa_acc:<12} {params:<12}")
    
    # Detailed comparison
    print(f"\n{'='*80}")
    print("DETAILED ANALYSIS")
    print(f"{'='*80}")
    
    # Find best model
    best_exp = None
    best_val_loss = float('inf')
    
    for exp in experiments:
        val_losses = exp.get('metrics', {}).get('val_loss', [])
        if val_losses and val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_exp = exp
    
    if best_exp:
        print(f"\n🏆 Best model (lowest val loss): {best_exp['exp_id']}")
        print(f"   Val loss: {best_val_loss:.4f}")
        
        # Compare to baseline
        baseline = next((e for e in experiments if 'baseline' in e.get('config', {}).get('model', '')), None)
        if baseline and best_exp != baseline:
            baseline_val = baseline.get('metrics', {}).get('val_loss', [])
            if baseline_val:
                improvement = (1 - best_val_loss / baseline_val[-1]) * 100
                if improvement > 0:
                    print(f"   ✅ {improvement:.1f}% better than baseline")
                else:
                    print(f"   ❌ {abs(improvement):.1f}% worse than baseline")


def plot_experiments(exp_ids: List[str] = None):
    """Generate plots comparing experiments."""
    if not HAS_MATPLOTLIB:
        print("Skipping plots (matplotlib not available)")
        return
    
    if exp_ids is None:
        exp_ids = ['exp001', 'exp002', 'exp003', 'exp004', 'exp005']
    
    experiments = []
    for exp_id in exp_ids:
        exp = load_experiment(f"experiments/{exp_id}")
        if exp and 'metrics' in exp:
            experiments.append(exp)
    
    if not experiments:
        print("No experiments found to plot")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Training loss
    ax = axes[0, 0]
    for exp in experiments:
        train_losses = exp['metrics'].get('train_loss', [])
        if train_losses:
            ax.plot(train_losses, label=exp['exp_id'], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation loss
    ax = axes[0, 1]
    for exp in experiments:
        val_losses = exp['metrics'].get('val_loss', [])
        if val_losses:
            ax.plot(val_losses, label=exp['exp_id'], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Train vs Val gap (overfitting indicator)
    ax = axes[1, 0]
    for exp in experiments:
        train_losses = exp['metrics'].get('train_loss', [])
        val_losses = exp['metrics'].get('val_loss', [])
        if train_losses and val_losses and len(train_losses) == len(val_losses):
            gaps = [v - t for t, v in zip(train_losses, val_losses)]
            ax.plot(gaps, label=exp['exp_id'], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val - Train Loss Gap')
    ax.set_title('Overfitting Indicator (lower is better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 4: Final validation loss comparison
    ax = axes[1, 1]
    exp_names = []
    final_vals = []
    for exp in experiments:
        val_losses = exp['metrics'].get('val_loss', [])
        if val_losses:
            exp_names.append(exp['exp_id'])
            final_vals.append(val_losses[-1])
    
    if final_vals:
        bars = ax.bar(exp_names, final_vals, color='steelblue', alpha=0.7)
        ax.set_ylabel('Final Validation Loss')
        ax.set_title('Final Validation Loss (lower is better)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, final_vals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('experiments/comparison.png', dpi=150, bbox_inches='tight')
    print("Saved plot to experiments/comparison.png")
    
    # Plot 5: Learning curves (log scale)
    fig, ax = plt.subplots(figsize=(10, 6))
    for exp in experiments:
        val_losses = exp['metrics'].get('val_loss', [])
        if val_losses:
            ax.semilogy(val_losses, label=exp['exp_id'], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss (log scale)')
    ax.set_title('Validation Loss (log scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('experiments/learning_curves.png', dpi=150, bbox_inches='tight')
    print("Saved plot to experiments/learning_curves.png")


def generate_summary():
    """Generate a summary markdown file."""
    exp_dirs = sorted(Path('experiments').glob('exp*'))
    
    if not exp_dirs:
        print("No experiments found")
        return
    
    summary_lines = [
        "# Experiment Results Summary\n",
        f"Generated: {Path.ctime(Path('experiments'))}\n",
        "## Experiments\n"
    ]
    
    for exp_dir in exp_dirs:
        exp = load_experiment(str(exp_dir))
        if not exp:
            continue
        
        summary_lines.append(f"### {exp['exp_id']}\n")
        
        if 'config' in exp:
            config = exp['config']
            summary_lines.append(f"- **Model**: {config.get('model', 'unknown')}")
            summary_lines.append(f"- **Latent steps**: {config.get('latent_steps', 0)}")
            summary_lines.append(f"- **Dimensions**: d_model={config.get('d_model', '?')}")
        
        if 'metrics' in exp:
            metrics = exp['metrics']
            val_losses = metrics.get('val_loss', [])
            if val_losses:
                summary_lines.append(f"- **Final val loss**: {val_losses[-1]:.4f}")
                summary_lines.append(f"- **Best val loss**: {min(val_losses):.4f}")
            
            eval_acc = metrics.get('eval_accuracy', [])
            if eval_acc:
                final_acc = eval_acc[-1].get('overall_accuracy', 0)
                summary_lines.append(f"- **QA accuracy**: {final_acc:.3f}")
        
        summary_lines.append("")
    
    # Write summary
    with open('experiments/SUMMARY.md', 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print("Generated experiments/SUMMARY.md")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument('--exp', type=str, help="Analyze specific experiment")
    parser.add_argument('--compare', nargs='+', help="Compare multiple experiments")
    parser.add_argument('--plot', action='store_true', help="Generate plots")
    parser.add_argument('--summary', action='store_true', help="Generate summary markdown")
    args = parser.parse_args()
    
    # Check if experiments directory exists
    if not Path('experiments').exists():
        print("No experiments directory found")
        print("Run experiments first with: python run_experiment.py ...")
        return
    
    if args.exp:
        # Analyze single experiment
        exp = load_experiment(f"experiments/{args.exp}")
        if exp:
            analyze_experiment(exp)
        else:
            print(f"Experiment {args.exp} not found")
    
    elif args.compare:
        # Compare experiments
        compare_experiments(args.compare)
    
    elif args.plot:
        # Generate plots
        plot_experiments()
    
    elif args.summary:
        # Generate summary
        generate_summary()
    
    else:
        # Analyze all experiments
        exp_dirs = sorted(Path('experiments').glob('exp*'))
        
        if not exp_dirs:
            print("No experiments found in experiments/")
            return
        
        print(f"Found {len(exp_dirs)} experiments\n")
        
        for exp_dir in exp_dirs:
            exp = load_experiment(str(exp_dir))
            if exp:
                analyze_experiment(exp)
        
        # Generate summary and plots
        generate_summary()
        if HAS_MATPLOTLIB:
            plot_experiments()


if __name__ == "__main__":
    main()
