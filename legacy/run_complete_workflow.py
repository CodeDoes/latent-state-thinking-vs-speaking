#!/usr/bin/env python3
"""
Complete automation for Kaggle experiment execution and analysis.

This script:
1. Monitors Kaggle notebook status
2. Downloads results when complete
3. Organizes files into experiments/ directory
4. Runs analysis
5. Generates final report

Usage:
    python run_complete_workflow.py
"""

import subprocess
import sys
import time
import json
import shutil
from pathlib import Path
from datetime import datetime

NOTEBOOK_ID = "kitastro/hybrid-latent-state-language-model"
OUTPUT_DIR = Path("kaggle_output")
LOG_FILE = Path("workflow.log")

def log(message):
    """Log message to console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    with open(LOG_FILE, 'a') as f:
        f.write(full_message + '\n')

def check_status():
    """Check Kaggle notebook status."""
    result = subprocess.run(
        ['kaggle', 'kernels', 'status', NOTEBOOK_ID],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

def download_results():
    """Download results from Kaggle."""
    log("📥 Downloading results...")
    
    # Clean old output
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir()
    
    # Download
    result = subprocess.run(
        ['kaggle', 'kernels', 'output', NOTEBOOK_ID, '-p', str(OUTPUT_DIR)],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        log("✅ Download complete")
        return True
    else:
        log(f"❌ Download failed: {result.stderr}")
        return False

def organize_files():
    """Organize downloaded files into experiments/ directory."""
    log("📂 Organizing files...")
    
    experiments_dir = Path("experiments")
    experiments_dir.mkdir(exist_ok=True)
    
    # Find and copy experiment directories
    for exp_dir in OUTPUT_DIR.glob("exp*"):
        if exp_dir.is_dir():
            dest = experiments_dir / exp_dir.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(exp_dir, dest)
            log(f"  ✓ {exp_dir.name}")
    
    # Copy JSON files
    for json_file in OUTPUT_DIR.glob("*.json"):
        shutil.copy(json_file, experiments_dir / json_file.name)
        log(f"  ✓ {json_file.name}")
    
    # Copy PNG files
    for png_file in OUTPUT_DIR.glob("*.png"):
        shutil.copy(png_file, experiments_dir / png_file.name)
        log(f"  ✓ {png_file.name}")
    
    log("✅ Files organized")

def run_analysis():
    """Run analysis script if it exists."""
    log("📊 Running analysis...")
    
    analysis_script = Path("analyze_results.py")
    if analysis_script.exists():
        result = subprocess.run(
            [sys.executable, str(analysis_script)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            log("✅ Analysis complete")
            log(f"\n{result.stdout}")
        else:
            log(f"⚠️  Analysis failed: {result.stderr}")
    else:
        log("⚠️  No analysis script found")

def generate_report():
    """Generate final summary report."""
    log("📝 Generating report...")
    
    results_file = Path("experiments/results.json")
    if not results_file.exists():
        log("❌ No results.json found")
        return
    
    with open(results_file) as f:
        results = json.load(f)
    
    report = []
    report.append("=" * 70)
    report.append("EXPERIMENT RESULTS SUMMARY")
    report.append("=" * 70)
    report.append("")
    
    # Find best model
    best_exp = None
    best_val_loss = float('inf')
    
    for exp_id, data in results.items():
        val_loss = data.get('val_loss', float('inf'))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_exp = exp_id
    
    # Print comparison
    report.append("Model Performance:")
    report.append("-" * 70)
    report.append(f"{'Exp':<10} {'Model':<20} {'Val Loss':<12} {'Params':<12}")
    report.append("-" * 70)
    
    for exp_id, data in results.items():
        model = data.get('model', 'unknown')
        val_loss = data.get('val_loss', 0)
        params = data.get('params', 0)
        marker = " 🏆" if exp_id == best_exp else ""
        report.append(f"{exp_id:<10} {model:<20} {val_loss:<12.4f} {params:<12,}{marker}")
    
    report.append("-" * 70)
    
    if best_exp:
        report.append(f"\n🏆 Best Model: {best_exp}")
        report.append(f"   Validation Loss: {best_val_loss:.4f}")
        
        # Compare to baseline
        baseline_loss = results.get('exp001', {}).get('val_loss', best_val_loss)
        if best_exp != 'exp001' and baseline_loss > 0:
            improvement = (1 - best_val_loss / baseline_loss) * 100
            if improvement > 0:
                report.append(f"   ✅ {improvement:.1f}% improvement over baseline")
            else:
                report.append(f"   ⚠️  {abs(improvement):.1f}% worse than baseline")
    
    # Check for QA results
    qa_file = Path("experiments/qa_results.json")
    if qa_file.exists():
        with open(qa_file) as f:
            qa_results = json.load(f)
        
        report.append("\n" + "=" * 70)
        report.append("QA EVALUATION RESULTS")
        report.append("=" * 70)
        report.append("")
        
        for exp_id, qa_data in qa_results.items():
            acc = qa_data.get('accuracy', 0)
            report.append(f"{exp_id}: {acc:.3f} accuracy")
            
            task_acc = qa_data.get('task_accuracy', {})
            for task, task_a in task_acc.items():
                report.append(f"  - {task}: {task_a:.3f}")
    
    report.append("\n" + "=" * 70)
    report.append("NEXT STEPS")
    report.append("=" * 70)
    report.append("")
    report.append("1. Review generated plots in experiments/")
    report.append("2. Check model samples in experiments/samples.json")
    report.append("3. Load best model: torch.load('experiments/{best_exp}/best_model.pt')")
    report.append("4. Run interactive testing with generate_sample() function")
    report.append("")
    report.append("=" * 70)
    
    # Write report
    report_text = '\n'.join(report)
    print(report_text)
    
    with open("FINAL_REPORT.txt", 'w') as f:
        f.write(report_text)
    
    log("✅ Report generated: FINAL_REPORT.txt")

def main():
    """Main workflow."""
    log("=" * 70)
    log("KAGGLE EXPERIMENT WORKFLOW")
    log("=" * 70)
    log(f"Notebook: {NOTEBOOK_ID}")
    log(f"URL: https://www.kaggle.com/code/{NOTEBOOK_ID}")
    log("")
    
    # Monitor until complete
    while True:
        status = check_status()
        log(f"Status: {status}")
        
        if 'complete' in status.lower():
            log("✅ Notebook completed successfully!")
            break
        elif 'error' in status.lower():
            log("❌ Notebook failed!")
            sys.exit(1)
        
        log("Waiting 5 minutes before next check...")
        time.sleep(300)  # 5 minutes
    
    # Download results
    if not download_results():
        sys.exit(1)
    
    # Organize files
    organize_files()
    
    # Run analysis
    run_analysis()
    
    # Generate report
    generate_report()
    
    log("")
    log("=" * 70)
    log("WORKFLOW COMPLETE")
    log("=" * 70)

if __name__ == "__main__":
    main()
