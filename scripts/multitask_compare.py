"""Aggregate multitask_analysis.json files. PYTHONPATH=. python scripts/multitask_compare.py runs/mt_p53 -> runs/mt_p53/multitask_summary.json
Runs the per-run analysis for any finished seed that lacks it."""
import sys, os, glob, json, subprocess
root = sys.argv[1]; rows = []
for d in sorted(glob.glob(f"{root}/*/s[0-9]*")):
    if not os.path.exists(f"{d}/final.npz"): continue
    if not os.path.exists(f"{d}/multitask_analysis.json"):
        subprocess.run([sys.executable, "scripts/multitask_analysis.py", d], stdout=subprocess.DEVNULL, check=True, env={**os.environ, "PYTHONPATH": "."})
    a = json.load(open(f"{d}/multitask_analysis.json"))
    for t, r in a["tasks_result"].items():
        rows.append(dict(cond=a["tasks"], task=t, seed=a["seed"], test_acc=r["test_acc"], key=r["key_freqs"], n_key=r["n_key"], full=r["full_loss"],
                         restricted=r["restricted_loss"], excluded=r["excluded_loss"], rand_min=r["random_restricted_min"], valid=r["valid"],
                         step50=r["first_step_test_acc_gt_0.5"]))
json.dump(rows, open(f"{root}/multitask_summary.json", "w"), indent=1)
for r in rows:
    print(f"{r['cond']:6s}{r['task']:4s} s{r['seed']} acc={r['test_acc']:.3f} n_key={r['n_key']} key={r['key']} full={r['full']:.2e} restr={r['restricted']:.2e} excl={r['excluded']:.1f} rand_min={r['rand_min']:.1f} valid={r['valid']} step50={r['step50']}")
