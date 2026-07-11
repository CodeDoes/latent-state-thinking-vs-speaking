#!/usr/bin/env python3
"""Download experiment results from Kaggle notebook output."""

import os
import sys
import subprocess
from pathlib import Path

def download_kaggle_output(notebook_path, output_dir="kaggle_output"):
    """
    Download output from a Kaggle notebook.
    
    Args:
        notebook_path: Full path to the notebook (e.g., 'username/notebook-name')
        output_dir: Local directory to save the output
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading output from {notebook_path}...")
    print(f"Saving to: {output_path.absolute()}")
    
    try:
        # Use Kaggle API to download output
        cmd = f"kaggle kernels output {notebook_path} -p {output_dir}"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        
        if result.returncode == 0:
            print("\n✅ Download complete!")
            print(f"\nFiles downloaded to {output_path}:")
            for item in sorted(output_path.rglob("*")):
                if item.is_file():
                    size = item.stat().st_size
                    print(f"  - {item.relative_to(output_path)} ({size:,} bytes)")
            return True
        else:
            print(f"\n❌ Download failed: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("\n❌ Kaggle CLI not found. Install with: pip install kaggle")
        print("   Then configure with: kaggle config path -p <path>")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def list_expected_outputs():
    """Show what files should be in the output."""
    print("\n📦 Expected output files after notebook runs:")
    print("\nModel files (in experiments/expXXX/):")
    print("  - best_model.pt - Best performing model weights")
    print("  - checkpoint.pt - Latest checkpoint with optimizer state")
    print("\nMetrics and samples:")
    print("  - results.json - Summary of all experiments")
    print("  - samples.json - Generated text samples")
    print("\nVisualizations:")
    print("  - loss_curves.png - Training/validation loss plots")
    print("  - final_comparison.png - Model comparison chart")
    print("\nTotal expected size: ~100-200 MB depending on model size")


def main():
    if len(sys.argv) < 2:
        print("Usage: python download_kaggle_results.py <username/notebook-name>")
        print("\nExample:")
        print("  python download_kaggle_results.py kitastro/hybrid-latent-state-language-model")
        print("\nTo find your notebook path:")
        print("  1. Go to your Kaggle notebook page")
        print("  2. Click 'Share' or look at the URL")
        print("  3. Path format: username/notebook-slug")
        list_expected_outputs()
        sys.exit(1)
    
    notebook_path = sys.argv[1]
    success = download_kaggle_output(notebook_path)
    
    if success:
        print("\n🎯 Next steps:")
        print("  1. Copy models to experiments/ folder:")
        print("     cp kaggle_output/experiments/* experiments/")
        print("  2. Analyze results:")
        print("     python analyze_results.py")
    else:
        print("\n💡 Alternative: Download manually from Kaggle")
        print("   1. Go to your notebook on Kaggle")
        print("   2. Click the 'Output' tab")
        print("   3. Click 'Download' to get a ZIP file")
        print("   4. Extract to kaggle_output/ folder")


if __name__ == "__main__":
    main()
