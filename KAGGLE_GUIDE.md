# Kaggle Experiment Guide

## Overview

This guide explains how to run the hybrid latent-state language model experiments on Kaggle GPU.

**Status**: All code validated on CPU. Ready for GPU execution.

## What's Ready

✅ **Architecture Improvements**
- All models produce `[batch, seq_len, vocab_size]` for fair comparison
- LatentSSM processes tokens sequentially (not mean pooling)
- Added `think_every` parameter to control thinking frequency
- Fixed tokenizer and dataset bugs

✅ **Models Implemented**
- BaselineTransformer: Standard causal LM
- LatentSSM: Sequential token processing + periodic thinking
- LatentSSMDecoder: SSM + multi-token decoder

✅ **Training Pipeline**
- Unified training loop for all models
- Temperature sampling for generation (not greedy)
- Improved QA evaluation with multiple samples per question
- Checkpointing and metrics tracking

✅ **Notebook**
- Self-contained `notebook.ipynb` with all 5 experiments
- GPU-ready with P100 compatibility check
- Includes improved evaluation code

## Experiments

| Exp | Model | Steps | Think Every | Purpose |
|-----|-------|-------|-------------|---------|
| 001 | Transformer baseline | 0 | - | Reference point |
| 002 | SSM (no thinking) | 0 | - | Pure recurrent baseline |
| 003 | SSM + thinking | 4 | 4 | Main hypothesis test |
| 004 | SSM + thinking | 4 | 8 | Test thinking frequency |
| 005 | SSM + decoder | 4 | 4 | Multi-token variant |

**Expected Runtime**: ~2-4 hours on Kaggle T4 GPU

## Manual Upload Instructions

Since the Kaggle CLI is not available in this environment, follow these steps:

### 1. Prepare Files

Ensure you have:
- `notebook.ipynb` (generated from `gen_notebook.py`)
- `kernel-metadata.json` (already configured)

### 2. Upload to Kaggle

**Option A: Web Interface**
1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Select "Import Notebook"
4. Upload `notebook.ipynb`
5. In Settings:
   - Enable GPU accelerator (T4 or P100)
   - Set to "Save version" after running
   - Enable internet access (for torch download if needed)

**Option B: Kaggle CLI (if available)**
```bash
pip install kaggle
kaggle kernels push -p .
```

### 3. Monitor Execution

- Check the notebook output in real-time
- Expected logs:
  - Dataset generation: ~5000 samples
  - Each experiment: 30 epochs
  - Metrics saved to `experiments/<exp_id>/`

### 4. Download Results

After execution completes:
1. Go to the notebook page
2. Click "Output" tab
3. Download the entire output directory
4. Extract to local `experiments/` folder

## Expected Results

### Metrics to Compare

1. **Validation Loss**: Does latent thinking improve generalization?
2. **QA Accuracy**: Can models answer reasoning questions?
   - Location tracking: "Where is X?"
   - Inventory tracking: "What does X have?"
   - Exact recall: "What is the secret code?"
3. **Training Speed**: How much overhead do thinking steps add?

### Hypothesis

**Primary**: Models with periodic thinking steps (exp003, exp004) will achieve lower validation loss than baseline (exp001) on reasoning tasks.

**Secondary**: 
- exp002 (SSM no thinking) tests if sequential processing helps
- exp004 (think every 8) tests if less frequent thinking is faster/better
- exp005 (decoder) tests multi-token generation

## Local Testing (Optional)

To test locally on CPU:

```bash
# Quick test (10 epochs, small model)
python run_experiment.py --exp_id test001 --model baseline --d_model 64 --epochs 2 --n_samples 100 --device cpu

# Full experiment (will be slow on CPU)
python run_experiment.py --exp_id exp001 --model baseline --d_model 256 --epochs 30 --device cpu
```

**CPU Timing** (for reference):
- Baseline: ~8s/epoch (100 samples, d_model=64)
- LatentSSM: ~106s/epoch (much slower due to sequential processing)
- On GPU, expect 10-50x speedup

## Troubleshooting

### P100 GPU Issues
If you get CUDA errors on P100:
- The notebook auto-detects P100 and installs torch 2.3.1+cu118
- If issues persist, try T4 GPU instead

### Out of Memory
If you hit OOM:
- Reduce `batch_size` in EXPS config
- Reduce sequence length (change `ml=128` to `ml=64` in TD dataset)

### Slow Training
- Latent models are inherently slower due to sequential processing
- This is expected and part of the hypothesis test
- On GPU, should still complete in reasonable time

## Next Steps After Results

1. **Analyze Results**
   - Compare validation loss curves
   - Check QA accuracy by task type
   - Look at sample generations

2. **Update PROGRESS.md**
   - Record final metrics
   - Document which hypothesis was confirmed
   - Save best model info

3. **Iterate**
   - If thinking helps: try more thinking steps
   - If thinking doesn't help: try different architectures
   - Move to Level 1 of research ladder

## File Structure

```
kaggle_pocalypse/
├── notebook.ipynb              # Main Kaggle notebook (self-contained)
├── kernel-metadata.json        # Kaggle kernel config
├── gen_notebook.py             # Generates notebook.ipynb
├── run_experiment.py           # Local experiment runner
├── src/
│   ├── models.py               # Model definitions
│   ├── trainer.py              # Training loop
│   ├── dataset.py              # Toy world generator
│   └── tokenizer.py            # Character tokenizer
├── experiments/                # Results (created during training)
│   ├── exp001/
│   │   ├── metrics.json
│   │   ├── best_model.pt
│   │   └── samples.txt
│   └── ...
└── results.json                # Summary of all experiments
```

## Contact

For questions about the experiment setup or results, refer to:
- `PROGRESS.md` - Current status and research log
- `PLAN.md` - Full research plan
- `AGENTS.md` - Project overview and architecture
