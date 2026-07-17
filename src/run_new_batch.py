"""Run all new-theory experiments at smoke scale (200 steps each, dim=32) to verify pipelines.

This is the 'ready to go' batch for the user request: more theories + experiments.

Usage:
  python -m src.run_new_batch --smoke   # 200 steps, dim=32, fast CPU
  python -m src.run_new_batch --full    # 2000 steps, dim=64, matches theory spec (takes longer)

Writes results to experiments/new_batch_summary.json
"""

import argparse, json, subprocess, sys
from pathlib import Path
import time

SMOKE_EXPS = [
    # injection-freq
    ["python", "-m", "src.train_injection_freq", "--fusion_mode", "front", "--exp_id", "inj_freq_front_smoke", "--steps", "200", "--dim", "32", "--log_every", "50"],
    ["python", "-m", "src.train_injection_freq", "--fusion_mode", "per_layer", "--exp_id", "inj_freq_perlayer_smoke", "--steps", "200", "--dim", "32", "--log_every", "50"],
    # dynamic patch
    ["python", "-m", "src.train_dynamic_patch", "--patch_mode", "fixed", "--exp_id", "dyn_patch_fixed_smoke", "--steps", "200", "--dim", "32", "--log_every", "50"],
    ["python", "-m", "src.train_dynamic_patch", "--patch_mode", "dynamic", "--threshold", "0.7", "--exp_id", "dyn_patch_dynamic_smoke", "--steps", "200", "--dim", "32", "--log_every", "50"],
    # token vs byte
    ["python", "-m", "src.train_token_byte", "--vocab_mode", "byte", "--exp_id", "tok_byte_bytehead_smoke", "--steps", "200", "--dim", "32", "--log_every", "50"],
    ["python", "-m", "src.train_token_byte", "--vocab_mode", "token", "--exp_id", "tok_byte_tokenhead_smoke", "--steps", "200", "--dim", "32", "--log_every", "50"],
    # rwkv carry
    ["python", "-m", "src.train_rwkv_carry", "--mode", "zero", "--exp_id", "rwkv_carry_zero_smoke", "--steps", "300", "--dim", "64", "--max_len", "128", "--log_every", "100"],
    ["python", "-m", "src.train_rwkv_carry", "--mode", "stateful", "--exp_id", "rwkv_carry_stateful_smoke", "--steps", "300", "--dim", "64", "--max_len", "128", "--log_every", "100"],
    # adaptive entropy sweep (just 3 points)
    ["python", "-m", "src.train_adaptive_entropy", "--entropy_weight", "0.0", "--exp_id", "adapt_ent_0.0_smoke", "--steps", "200", "--dim", "32", "--log_every", "50"],
    ["python", "-m", "src.train_adaptive_entropy", "--entropy_weight", "0.01", "--exp_id", "adapt_ent_0.01_smoke", "--steps", "200", "--dim", "32", "--log_every", "50"],
    ["python", "-m", "src.train_adaptive_entropy", "--entropy_weight", "0.1", "--exp_id", "adapt_ent_0.1_smoke", "--steps", "200", "--dim", "32", "--log_every", "50"],
]

FULL_EXPS = [
    ["python", "-m", "src.train_injection_freq", "--fusion_mode", "front", "--exp_id", "inj_freq_front_001", "--steps", "2000"],
    ["python", "-m", "src.train_injection_freq", "--fusion_mode", "per_layer", "--exp_id", "inj_freq_perlayer_001", "--steps", "2000"],
    ["python", "-m", "src.train_dynamic_patch", "--patch_mode", "fixed", "--exp_id", "dyn_patch_fixed_001", "--steps", "2000"],
    ["python", "-m", "src.train_dynamic_patch", "--patch_mode", "dynamic", "--threshold", "0.7", "--exp_id", "dyn_patch_dynamic_07_001", "--steps", "2000"],
    ["python", "-m", "src.train_token_byte", "--vocab_mode", "byte", "--exp_id", "tok_byte_bytehead_001", "--steps", "2000"],
    ["python", "-m", "src.train_token_byte", "--vocab_mode", "token", "--exp_id", "tok_byte_tokenhead_001", "--steps", "2000"],
    ["python", "-m", "src.train_rwkv_carry", "--mode", "zero", "--exp_id", "rwkv_carry_zero_001", "--steps", "2000", "--max_len", "256", "--noise_max", "10"],
    ["python", "-m", "src.train_rwkv_carry", "--mode", "stateful", "--exp_id", "rwkv_carry_stateful_001", "--steps", "2000", "--max_len", "256", "--noise_max", "10"],
    ["python", "-m", "src.train_adaptive_entropy", "--sweep", "--exp_id_prefix", "adapt_ent", "--steps", "2000"],
]

def run_commands(cmds):
    results=[]
    for cmd in cmds:
        print(f"\n{'='*70}\n>>> {' '.join(cmd)}\n{'='*70}")
        t0=time.time()
        try:
            proc=subprocess.run(cmd, capture_output=False, text=True, timeout=600)
            ok=proc.returncode==0
        except Exception as e:
            print(f"FAILED: {e}")
            ok=False
        elapsed=time.time()-t0
        # read metrics if exists
        exp_id=None
        if "--exp_id" in cmd:
            idx=cmd.index("--exp_id")
            exp_id=cmd[idx+1]
        # for sweep, exp_id_prefix case
        metrics=None
        if exp_id:
            mp=Path(f"experiments/{exp_id}/metrics.json")
            if mp.exists():
                try:
                    metrics=json.loads(mp.read_text())
                except: metrics=None
        results.append({"cmd":" ".join(cmd),"exp_id":exp_id,"ok":ok,"elapsed":elapsed,"metrics":metrics})
        # save incremental
        Path("experiments/new_batch_summary.json").write_text(json.dumps(results,indent=2))
    return results

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="smoke 200 steps each")
    ap.add_argument("--full", action="store_true", help="full 2000 steps")
    args=ap.parse_args()
    if not args.smoke and not args.full:
        args.smoke=True

    cmds = SMOKE_EXPS if args.smoke else FULL_EXPS
    print(f"Running {len(cmds)} experiments, mode={'smoke' if args.smoke else 'full'}")
    results=run_commands(cmds)
    print("\n=== SUMMARY ===")
    for r in results:
        print(f"{r['exp_id'] or r['cmd']}: ok={r['ok']} elapsed={r['elapsed']:.1f}s metrics={r['metrics']}")
    Path("experiments/new_batch_summary.json").write_text(json.dumps(results,indent=2))
    print(f"\nSaved experiments/new_batch_summary.json")

if __name__=="__main__":
    main()
