"""Resumable multi-seed training driver. Safe to call repeatedly: finished seeds are skipped and
unfinished ones resume. Stops cleanly when --budget seconds are used.
  python scripts/sweep.py --p 53 --train_frac 0.5 --steps 6000 --seeds 0-29 --out runs/sweep_p53 --budget 520
"""
import argparse, os, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--p", type=int, required=True)
ap.add_argument("--train_frac", type=float, default=0.5)
ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--seeds", default="0-9")
ap.add_argument("--out", required=True)
ap.add_argument("--budget", type=float, default=520)
ap.add_argument("--ckpt_every", type=int, default=250)
a = ap.parse_args()
lo, hi = map(int, a.seeds.split("-")); t0 = time.time()
for s in range(lo, hi + 1):
    d = f"{a.out}/s{s}"
    if os.path.exists(f"{d}/final.npz"):
        continue
    left = a.budget - (time.time() - t0)
    if left < 30:
        break
    cmd = [sys.executable, "-m", "tiny_circuits.train", "--p", str(a.p), "--seed", str(s), "--train_frac", str(a.train_frac),
           "--steps", str(a.steps), "--log_every", "250", "--ckpt_every", str(a.ckpt_every), "--max_seconds", str(left), "--out", d]
    if os.path.exists(f"{d}/state.pkl"):
        cmd.append("--resume")
    subprocess.run(cmd, stdout=subprocess.DEVNULL, check=True, env={**os.environ, "PYTHONPATH": "."})
done = [s for s in range(lo, hi + 1) if os.path.exists(f"{a.out}/s{s}/final.npz")]
print(f"finished {len(done)}/{hi - lo + 1} seeds; elapsed {time.time() - t0:.0f}s")
