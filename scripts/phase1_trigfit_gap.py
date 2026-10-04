"""Where does the logit energy of the Phase 1 model live in the 3-D Fourier basis over (a, b, c)?
  PYTHONPATH=. python scripts/phase1_trigfit_gap.py runs/p113_s0   -> <run>/trigfit_gap.json
Because the basis is orthonormal, summing the energy of a subset of components equals the least-squares fit onto that subset."""
import sys, json
import numpy as np
from tiny_circuits import analysis as A

run = sys.argv[1]; cfg = json.load(open(f"{run}/config.json")); p = cfg["p"]
P = A.load(f"{run}/final.npz"); key = [int(k) for k in A.key_frequencies(P, p)]
F, names = A.fourier_basis(p); fr = A.freq_of_index(p)
L = np.asarray(A.full_logits(P, p), dtype=np.float64)
T = np.einsum("ia,jb,kc,abc->ijk", F, F, F, L, optimize=True)
E = T ** 2
fi, fj, fk = fr[:, None, None], fr[None, :, None], fr[None, None, :]
isk = lambda f: np.isin(f, key)
cvar = fk > 0                                   # components that vary with the class c (others cannot change the argmax)
tot = E[np.broadcast_to(cvar, E.shape)].sum()
groups = {
    "class_bias (0,0,k)": (fi == 0) & (fj == 0) & cvar,
    "same_freq_kkk (all 8 sign patterns)": isk(fi) & (fi == fj) & (fj == fk),
    "kk_l: a,b at key k, c at a different key l": isk(fi) & (fi == fj) & isk(fk) & (fk != fi),
    "kk_l: a,b at key k, c at non-key l": isk(fi) & (fi == fj) & ~isk(fk) & cvar,
    "a,b at different key freqs": isk(fi) & isk(fj) & (fi != fj) & cvar,
    "one of a,b constant, other at key freq": (((fi == 0) & isk(fj)) | ((fj == 0) & isk(fi))) & cvar,
}
res = {"key_freqs": key, "class_varying_energy_total": float(tot)}
used = np.zeros(E.shape, bool)
for g, m in groups.items():
    m = np.broadcast_to(m & cvar, E.shape) & ~used; used |= m
    res[g] = float(E[m].sum() / tot)
res["everything else"] = float(E[np.broadcast_to(cvar, E.shape) & ~used].sum() / tot)

# sign-pattern view of the same-frequency block: for each key k, energy by (a-type, b-type, c-type) in {cos, sin}
def kind(idx): return "const" if idx == 0 else ("cos" if idx % 2 == 1 else "sin")
blk = {}
for k in key:
    ix = [2 * k - 1, 2 * k]                       # cos_k, sin_k rows
    for i in ix:
        for j in ix:
            for l in ix:
                blk[f"k{k}:{kind(i)}_a*{kind(j)}_b*{kind(l)}_c"] = float(E[i, j, l] / tot)
res["same_freq_components"] = dict(sorted(blk.items(), key=lambda kv: -kv[1])[:16])

# reconstruction quality (argmax accuracy and CE) as component groups are added
toks, labels = A.all_pairs(p)
def recon(mask):
    return np.einsum("ia,jb,kc,ijk->abc", F, F, F, T * mask, optimize=True)
def score(mask):
    R = recon(mask).reshape(p * p, p)
    return dict(argmax_acc=float((R.argmax(1) == labels).mean()), ce=A.ce(__import__("jax.numpy", fromlist=["x"]).asarray(R), labels))
bias = np.broadcast_to(groups["class_bias (0,0,k)"], E.shape)
kkk = np.broadcast_to(groups["same_freq_kkk (all 8 sign patterns)"], E.shape)
cross = kkk | np.broadcast_to(groups["kk_l: a,b at key k, c at a different key l"], E.shape)
# the textbook 2-component form, for reference: cos/sin of w(a+b-c) only
res["recon_bias_only"] = score(bias)
res["recon_bias_plus_all_kkk"] = score(bias | kkk)
res["recon_bias_plus_kkk_plus_cross_kkl"] = score(bias | cross)
# greedy: add largest class-varying components until argmax accuracy >= 99% (reports how many are needed)
order = np.argsort(-(E * np.broadcast_to(cvar, E.shape)).ravel())
for n in (8, 16, 32, 64, 128, 256, 512, 1024):
    m = np.zeros(E.size, bool); m[order[:n]] = True
    res[f"greedy_top_{n}_components"] = score(m.reshape(E.shape))
res["top_components_outside_bias_and_kkk"] = []
for flat in order[:400]:
    i, j, k = np.unravel_index(flat, E.shape)
    if bias[i, j, k] or kkk[i, j, k]: continue
    res["top_components_outside_bias_and_kkk"].append(dict(a=f"{names[i]}", b=f"{names[j]}", c=f"{names[k]}", frac=float(E[i, j, k] / tot)))
    if len(res["top_components_outside_bias_and_kkk"]) >= 12: break
json.dump(res, open(f"{run}/trigfit_gap.json", "w"), indent=1)
print(json.dumps(res, indent=1))
