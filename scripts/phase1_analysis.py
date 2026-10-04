"""Phase 1 analysis: confirm the Fourier circuit in a grokked mod-p addition model.
  python scripts/phase1_analysis.py runs/p113_s0
Writes <run>/analysis.json and <run>/progress.json."""
import sys, json, glob, os
import numpy as np, jax.numpy as jnp
from tiny_circuits import analysis as A
from tiny_circuits.data import modular_addition

run = sys.argv[1]
cfg = json.load(open(f"{run}/config.json")); p = cfg["p"]
train, test = modular_addition(p, cfg["train_frac"], cfg["seed"] if cfg.get("data_seed") is None else cfg["data_seed"])
params = A.load(f"{run}/final.npz")
out = {}

# 1. key frequencies from the embedding
norms = A.embed_fourier_norms(params, p)
keyf = A.key_frequencies(params, p)
embed_top = [int(k) for k in np.argsort(-norms[1:])[:len(keyf)] + 1]
out["key_freqs_agree_with_embedding_top_k"] = sorted(embed_top) == sorted(int(k) for k in keyf)
out["embed_freq_norms_top"] = {int(k): float(norms[k]) for k in np.argsort(-norms[1:])[:10] + 1}
out["key_freqs"] = [int(k) for k in keyf]
out["embed_norm_share_in_key"] = float((norms[keyf] ** 2).sum() / (norms[1:] ** 2).sum())

# 2. restricted / excluded loss on train and test
for name, sp in (("train", train), ("test", test)):
    full, res, exc = A.restricted_excluded_loss(params, p, keyf, split=sp)
    out[f"loss_{name}"] = dict(full=full, restricted=res, excluded=exc)

# 2b. control: same-size random frequency sets (excluding the key ones)
rng = np.random.RandomState(0); others = [k for k in range(1, (p - 1) // 2 + 1) if k not in set(keyf)]
ctrl = []
for _ in range(20):
    fs = rng.choice(others, len(keyf), replace=False)
    ctrl.append(A.restricted_excluded_loss(params, p, fs, split=test)[1:])
ctrl = np.array(ctrl)
out["control_random_freqs_test"] = dict(restricted_mean=float(ctrl[:, 0].mean()), restricted_min=float(ctrl[:, 0].min()),
                                        excluded_mean=float(ctrl[:, 1].mean()), excluded_max=float(ctrl[:, 1].max()))

# 3. head ablations and neuron structure
out["head_ablation_test_loss"] = A.head_ablation(params, p, test)
dom, frac = A.neuron_frequency_profile(params, p)
out["neuron_dominant_freq_counts"] = {int(k): int((dom == k).sum()) for k in np.unique(dom)}
out["neuron_median_explained_frac"] = float(np.median(frac))
out["neuron_frac_explained>0.9"] = float((frac > 0.9).mean())
order = list(np.argsort(-frac))
out["neuron_ablation_test_loss(strongest-first)"] = A.neuron_ablation_curve(params, p, test, order)
json.dump(out, open(f"{run}/analysis.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))

# 4. progress measures over checkpoints
prog = []
for f in sorted(glob.glob(f"{run}/ckpt_*.npz")):
    step = int(os.path.basename(f)[5:11]); pr = A.load(f)
    r = A.restricted_excluded_loss(pr, p, keyf, split=test)
    t = A.restricted_excluded_loss(pr, p, keyf, split=train)
    prog.append(dict(step=step, test_full=r[0], test_restricted=r[1], test_excluded=r[2],
                     train_full=t[0], train_restricted=t[1], train_excluded=t[2]))
json.dump(prog, open(f"{run}/progress.json", "w"))
print("progress points:", len(prog))
