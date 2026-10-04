"""Per-run analysis for multitask runs (Addendum C). PYTHONPATH=. python scripts/multitask_analysis.py <run_dir> [weights.npz]
Writes <run_dir>/multitask_analysis.json. Additive: Z_p basis. Multiplicative: discrete-log coordinates, Z_{p-1} basis, nonzero a,b,c only."""
import sys, json, itertools
import numpy as np, jax.numpy as jnp
from tiny_circuits import analysis as A
from tiny_circuits.model import forward
from tiny_circuits import multitask as M

run = sys.argv[1]; wfile = sys.argv[2] if len(sys.argv) > 2 else "final.npz"
cfg = json.load(open(f"{run}/config.json")); p, tasks, seed, frac = cfg["p"], cfg["tasks"], cfg["seed"], cfg["train_frac"]
tl = ["add", "mul"] if tasks == "addmul" else [tasks]
P = A.load(f"{run}/{wfile}")
_, test, _ = M.make_data(p, tl, frac, seed)
g, exp, log = M.log_tables(p); n = p - 1
rng = np.random.RandomState(0)
out = dict(run=run, weights=wfile, p=p, tasks=tasks, seed=seed, train_frac=frac)
log_json = json.load(open(f"{run}/log.json"))

def logits_all(t):
    tok, lab = M.task_pairs(p, t)
    return np.asarray(forward(P, jnp.asarray(tok)), dtype=np.float64).reshape(p, p, p), tok, lab

def controls(L, nn, key, ia, ib, labels, ndraw=20):
    pool = [k for k in range(1, (nn - 1) // 2 + 1 if nn % 2 else nn // 2 + 1) if k not in key]
    if len(key) > len(pool): return dict(random_restricted_min=float('nan'), random_restricted_mean=float('nan'))
    rs = [M.restricted_excluded_generic(L, nn, list(rng.choice(pool, len(key), replace=False)), ia, ib, labels)[1] for _ in range(ndraw)]
    return dict(random_restricted_min=float(min(rs)), random_restricted_mean=float(np.mean(rs)))

res = {}
for t in tl:
    L, tok, lab = logits_all(t); xt, yt = test[t]
    acc = float((L.reshape(-1, p)[np.ravel_multi_index((xt[:, 0], xt[:, 1]), (p, p))].argmax(1) == yt).mean())
    r = dict(test_acc=acc)
    if t == "add":
        Lc, nn, ia, ib, labels = L, p, xt[:, 0], xt[:, 1], yt
    else:
        Lc = L[np.ix_(exp, exp, exp)]; nn = n                      # [log a, log b, log c]
        m = (xt[:, 0] != 0) & (xt[:, 1] != 0)
        ia, ib, labels = log[xt[m, 0]], log[xt[m, 1]], log[yt[m]]
        sub = Lc[ia, ib].argmax(1) == labels
        r["test_acc_nonzero_pairs_over_52_classes"] = float(sub.mean())
    key = M.key_freqs_generic(Lc, nn)
    full, restr, excl = M.restricted_excluded_generic(Lc, nn, key, ia, ib, labels)
    r.update(key_freqs=key, n_key=len(key), full_loss=full, restricted_loss=restr, excluded_loss=excl, **controls(Lc, nn, key, ia, ib, labels))
    # first logged step with test acc > 0.5
    r["first_step_test_acc_gt_0.5"] = next((int(x["step"]) for x in log_json if x[f"{t}_test_acc"] > 0.5), None)
    r["valid"] = False if r["random_restricted_min"] != r["random_restricted_min"] else bool((restr < 10 * full or restr < 1e-3) and excl > 5 and r["random_restricted_min"] > 5 * max(full, 1e-3)) if len(key) else False
    res[t] = r; res[t]["_Lc"] = None
    if t == "add": Ladd_key = key
    else: Lmul_key = key
for t in res: res[t].pop("_Lc")
out["tasks_result"] = res

if tasks == "addmul":
    E = np.asarray(P["W_E"], dtype=np.float64)[1:p]                      # rows a = 1..p-1
    a = np.arange(1, p); la = log[a]
    def feats(freqs, coord, nn):
        return np.stack([f(2 * np.pi * k * coord / nn) for k in freqs for f in (np.cos, np.sin)]) if len(freqs) else np.zeros((0, len(coord)))
    def overlap(Fa, Fm):
        U, V = Fa @ E, Fm @ E
        U /= np.linalg.norm(U, axis=1, keepdims=True); V /= np.linalg.norm(V, axis=1, keepdims=True)
        return float(((U @ V.T) ** 2).mean())
    Fa = feats(Ladd_key, a, p)
    pool = [k for k in range(1, n // 2 + 1) if k not in Lmul_key]
    if len(Lmul_key) > len(pool): out["M3"] = None   # non-grokked model: too many "key" frequencies for a null
    else:
        O_real = overlap(Fa, feats(Lmul_key, la, n))
        nulls = [overlap(Fa, feats(list(rng.choice(pool, len(Lmul_key), replace=False)), la, n)) for _ in range(200)]
        out["M3"] = dict(O_real=O_real, O_null_mean=float(np.mean(nulls)), ratio=O_real / float(np.mean(nulls)))
    # M5: neuron specialisation by task
    ms = {}
    for t in tl:
        tok, _ = M.task_pairs(p, t)
        _, c = forward(P, jnp.asarray(tok), return_cache=True)
        ms[t] = np.asarray((c["mlp_post"][:, -1, :] ** 2).mean(0), dtype=np.float64)
    ratio = (ms["add"] + 1e-12) / (ms["mul"] + 1e-12)
    out["M5"] = dict(frac_neurons_ratio_ge_5_either_way=float(((ratio >= 5) | (ratio <= 0.2)).mean()),
                     frac_add_selective=float((ratio >= 5).mean()), frac_mul_selective=float((ratio <= 0.2).mean()),
                     frac_dead=float(((ms["add"] < 1e-9) & (ms["mul"] < 1e-9)).mean()))
json.dump(out, open(f"{run}/multitask_analysis.json", "w"), indent=1)
print(json.dumps(out, indent=1))
