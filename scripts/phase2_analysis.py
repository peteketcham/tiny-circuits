"""Per-seed circuit summary for a sweep directory.
  python scripts/phase2_analysis.py runs/sweep_p53 [--min_frac 0.01] [--tag t01]
Writes <dir>/summary_<tag>.json (one record per seed) and prints a table."""
import sys, json, glob, os, argparse
import numpy as np
from tiny_circuits import analysis as A
from tiny_circuits.data import modular_addition

ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--min_frac", type=float, default=0.01)
ap.add_argument("--grok_thresh", type=float, default=0.99)
ap.add_argument("--tag", default="main"); a = ap.parse_args()
rows = []
for d in sorted(glob.glob(f"{a.dir}/s[0-9]*"), key=lambda x: int(x.rsplit("/s", 1)[1])):
    if not os.path.exists(f"{d}/final.npz"):
        continue
    cfg = json.load(open(f"{d}/config.json")); p = cfg["p"]; seed = cfg["seed"]
    log = json.load(open(f"{d}/log.json"))
    r = dict(seed=seed, final_test_acc=log[-1]["test_acc"], final_train_acc=log[-1]["train_acc"])
    onset = [x["step"] for x in log if x["test_acc"] > 0.5]
    r["grok_step_50"] = onset[0] if onset else None
    P = A.load(f"{d}/final.npz")
    if r["final_test_acc"] < a.grok_thresh:
        r["status"] = "no_grok"; rows.append(r); continue
    r["status"] = "grokked"
    train, test = modular_addition(p, cfg["train_frac"], seed)
    kf = [int(k) for k in A.key_frequencies(P, p, min_frac=a.min_frac)]
    r["key_freqs"] = kf
    full, res, exc = A.restricted_excluded_loss(P, p, kf, split=test)
    r["test_loss"] = dict(full=full, restricted=res, excluded=exc)
    rng = np.random.RandomState(seed); others = [k for k in range(1, (p - 1) // 2 + 1) if k not in kf]
    ctrl = [A.restricted_excluded_loss(P, p, rng.choice(others, len(kf), replace=False), split=test)[1:] for _ in range(10)]
    ctrl = np.array(ctrl); r["control"] = dict(restricted_min=float(ctrl[:, 0].min()), excluded_max=float(ctrl[:, 1].max()))
    en = A.embed_fourier_norms(P, p); r["embed_top_n"] = sorted(int(k) for k in np.argsort(-en[1:])[:len(kf)] + 1)
    # heads: dominant frequency of each head's OV output and its share
    F, _ = A.fourier_basis(p); fr = A.freq_of_index(p); heads = []
    for h in range(P["W_Q"].shape[0]):
        M = np.asarray(P["W_E"])[:p] @ np.asarray(P["W_V"][h]) @ np.asarray(P["W_O"][h]); pr = F @ M
        n = np.array([np.sqrt((pr[fr == k] ** 2).sum()) for k in range((p - 1) // 2 + 1)])
        tot = (n[kf] ** 2).sum() + 1e-12; top = max(kf, key=lambda k: n[k]); heads.append((int(top), float(n[top] ** 2 / tot)))
    r["heads_dominant_freq_share"] = heads
    # early prediction: embedding top-n at last checkpoint with test acc < 0.2
    steps = {x["step"]: x["test_acc"] for x in log}
    early = [s for s in sorted(steps) if steps[s] < 0.2]
    if early:
        s_early = max(s for s in early if os.path.exists(f"{d}/ckpt_{s:06d}.npz")) if any(os.path.exists(f"{d}/ckpt_{s:06d}.npz") for s in early) else None
        if s_early is not None and s_early > 0:
            Pe = A.load(f"{d}/ckpt_{s_early:06d}.npz"); ee = A.embed_fourier_norms(Pe, p)
            pred = set(int(k) for k in np.argsort(-ee[1:])[:len(kf)] + 1); fin = set(kf)
            r["early_step"] = s_early; r["early_pred_jaccard"] = len(pred & fin) / len(pred | fin)
    rows.append(r)
json.dump(rows, open(f"{a.dir}/summary_{a.tag}.json", "w"), indent=1, default=float)
g = [r for r in rows if r["status"] == "grokked"]
print(f"{len(rows)} seeds, {len(g)} grokked")
for r in rows:
    if r["status"] != "grokked":
        print(f"s{r['seed']:>2} NO GROK test_acc={r['final_test_acc']:.3f}"); continue
    t = r["test_loss"]
    print(f"s{r['seed']:>2} grok@{r['grok_step_50']} freqs={r['key_freqs']} full={t['full']:.1e} restr={t['restricted']:.1e} excl={t['excluded']:.1f} "
          f"ctrl(restr_min={r['control']['restricted_min']:.1f}) early_J={r.get('early_pred_jaccard', float('nan')):.2f}")
