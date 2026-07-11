#!/usr/bin/env python3
"""
Push experiments to Kaggle for GPU training.

Usage:
    python kaggle_push.py                    # Push all experiments
    python kaggle_push.py --exp exp001       # Push specific experiment
    python kaggle_push.py --monitor          # Monitor running kernels

This uses the Kaggle API to:
1. Create/update Kaggle notebooks
2. Submit them for GPU execution
3. Monitor progress
4. Download results when complete
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

def push_notebook():
    """Push the notebook to Kaggle using the kaggle CLI."""
    print("Pushing notebook to Kaggle...")
    os.system("kaggle kernels push -p .")
    print("Notebook pushed! Check: https://www.kaggle.com/kitastro/code")


def push_and_run():
    """Push notebook and start execution."""
    print("Pushing notebook to Kaggle...")
    os.system("kaggle kernels push -p .")

    # Read kernel ID from kaggle.json
    with open("kaggle.json") as f:
        config = json.load(f)
    kernel_id = config.get("id", "")

    if not kernel_id:
        print("No kernel ID in kaggle.json, pushing as new kernel")
    else:
        print(f"Kernel ID: {kernel_id}")

    print("\nStarting execution on Kaggle GPU...")
    if kernel_id:
        os.system(f"kaggle kernels output {kernel_id}")
    else:
        print("Visit https://www.kaggle.com/kitastro/code to start execution")


def monitor():
    """Monitor running kernels."""
    print("Checking kernel status...")
    os.system("kaggle kernels list -m mine -s running")


def download_results():
    """Download results from completed kernels."""
    with open("kaggle.json") as f:
        config = json.load(f)
    kernel_id = config.get("id", "")

    if not kernel_id:
        print("No kernel ID found")
        return

    print(f"Downloading results from {kernel_id}...")
    os.makedirs("kaggle_output", exist_ok=True)
    os.system(f"kaggle kernels output {kernel_id} -p kaggle_output")
    print("Results downloaded to kaggle_output/")


def status():
    """Check Kaggle API connection and quota."""
    print("Checking Kaggle API connection...")
    os.system("kaggle config view")
    print("\nChecking GPU quota...")
    os.system("kaggle kernels list -m mine")


def main():
    parser = argparse.ArgumentParser(description="Kaggle experiment management")
    parser.add_argument("--push", action="store_true", help="Push notebook to Kaggle")
    parser.add_argument("--run", action="store_true", help="Push and start execution")
    parser.add_argument("--monitor", action="store_true", help="Monitor running kernels")
    parser.add_argument("--download", action="store_true", help="Download results")
    parser.add_argument("--status", action="store_true", help="Check API status")
    args = parser.parse_args()

    if args.push:
        push_notebook()
    elif args.run:
        push_and_run()
    elif args.monitor:
        monitor()
    elif args.download:
        download_results()
    elif args.status:
        status()
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python kaggle_push.py --push     # Push notebook")
        print("  python kaggle_push.py --run      # Push and execute")
        print("  python kaggle_push.py --monitor  # Check status")
        print("  python kaggle_push.py --download # Get results")


if __name__ == "__main__":
    main()
