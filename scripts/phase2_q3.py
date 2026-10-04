"""Q3 (PLAN.md): can the final key frequencies be predicted from the embedding before the model generalises?
Pre-registered rule: take the last checkpoint where test acc < 20%; take the top-n embedding frequencies there
(n = size of the final key set); score Jaccard with the final key set. Chance is estimated by simulation.
Also (POST HOC, labelled): mean Jaccard at fixed steps.
  python scripts/phase2_q3.py runs/sweep_p53 runs/sweep_p53_early"""
import sys, json, glob, os
import numpy as np, jax.numpy as jnp
from tiny_circuits import analysis as A
from tiny_circuits.model import forward
from tiny_circuits.data import modular_addition

main, early = sys.argv[1], sys.argv[2]
rows = [r for r in json.load(open(f"{main}/summary_main.json")) if r["status"] == "grokked"]
rng = np.random.RandomState(0); out_rows, curve = [], {}
for r in rows:
    s = r["seed"]; cfg = json.load(open(f"{early}/s{s}/config.json")); p = cfg["p"]
    _, (xt, yt) = modular_addition(p, cfg["train_frac"], s); xt, yt = jnp.asarray(xt), np.asarray(yt)
    fin, n = set(r["key_freqs"]), len(r["key_freqs"])
    steps = sorted(int(os.path.basename(f)[5:11]) for f in glob.glob(f"{early}/s{s}/ckpt_*.npz"))
    acc, jac = {}, {}
    for st in steps:
        P = A.load(f"{early}/s{s}/ckpt_{st:06d}.npz")
        acc[st] = float((np.asarray(forward(P, xt)).argmax(-1) == yt).mean())
        pred = set(int(k) for k in np.argsort(-A.embed_fourier_norms(P, p)[1:])[:n] + 1)
        jac[st] = len(pred & fin) / len(pred | fin)
    below = [st for st in steps if acc[st] < 0.2]
    s_early = max(below)
    def _j():
        q = set(rng.choice(np.arange(1, (p - 1) // 2 + 1), n, replace=False)); return len(q & fin) / len(q | fin)
    chance = np.mean([_j() for _ in range(2000)])
    out_rows.append(dict(seed=s, n=n, early_step=s_early, acc_at_early=acc[s_early], jaccard=jac[s_early], chance=float(chance),
                         first_step_acc_ge_20=min([st for st in steps if acc[st] >= 0.2], default=None)))
    for st in steps: curve.setdefault(st, []).append(jac[st])
J = np.array([x["jaccard"] for x in out_rows]); C = np.array([x["chance"] for x in out_rows])
res = dict(n_seeds=len(out_rows), mean_jaccard=float(J.mean()), median_jaccard=float(np.median(J)), mean_chance=float(C.mean()),
           frac_seeds_above_0p5=float((J > 0.5).mean()), prediction_holds=bool(J.mean() > 0.5),
           median_early_step=float(np.median([x["early_step"] for x in out_rows])),
           POSTHOC_mean_jaccard_by_step={int(st): float(np.mean(v)) for st, v in curve.items() if st in (0, 50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500)},
           per_seed=out_rows)
json.dump(res, open(f"{main}/q3.json", "w"), indent=1)
print({k: v for k, v in res.items() if k != "per_seed"})
