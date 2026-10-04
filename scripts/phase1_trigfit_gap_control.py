"""Exploratory test + random control for the trig-fit gap (README claim 10).
Question: with a, b restricted to the key frequencies, how many argmax decisions are recovered if the output axis c may use
(i) key frequencies only, (ii) key + harmonics/combinations (m*k, m=2,3,4; k1+-k2), (iii) key + the same number of random frequencies?
  PYTHONPATH=. python scripts/phase1_trigfit_gap_control.py runs/p113_s0  -> <run>/trigfit_gap_control.json"""
import sys, json, itertools
import numpy as np
from tiny_circuits import analysis as A

run = sys.argv[1]; cfg = json.load(open(f"{run}/config.json")); p = cfg["p"]
P = A.load(f"{run}/final.npz"); key = [int(k) for k in A.key_frequencies(P, p)]
F, _ = A.fourier_basis(p); fr = A.freq_of_index(p)
L = np.asarray(A.full_logits(P, p), dtype=np.float64)
Tab = np.einsum("ia,jb,abc->ijc", F, F, L, optimize=True)
mkab = np.isin(fr, key)[:, None] & np.isin(fr, key)[None, :] & (fr[:, None] == fr[None, :])
mc00 = np.zeros((p, p), bool); mc00[0, 0] = True
_, labels = A.all_pairs(p)
Tc = np.einsum("kc,ijc->ijk", F, Tab, optimize=True)

def acc_for(cfreqs):
    ck = np.isin(fr, list(cfreqs))
    M = (mkab[:, :, None] & ck[None, None, :]) | (mc00[:, :, None] & (fr > 0)[None, None, :])
    R = np.einsum("ia,jb,kc,ijk->abc", F, F, F, Tc * M, optimize=True).reshape(p * p, p)
    return float((R.argmax(1) == labels).mean())

fold = lambda f: min(f % p, (-f) % p)
struct = set(key)
for k in key:
    for m in (2, 3, 4): struct.add(fold(m * k))
for k1, k2 in itertools.combinations(key, 2):
    for s in (1, -1): struct.add(fold(k1 + s * k2))
n_extra = len(struct - set(key))
rng = np.random.RandomState(0); non = [f for f in range(1, (p - 1) // 2 + 1) if f not in key]
accs = [acc_for(set(key) | set(rng.choice(non, n_extra, replace=False))) for _ in range(40)]
# c-axis spectrum of the (a,b at key k) block
spec = np.array([(Tc[:, :, fr == l] ** 2)[mkab].sum() for l in range((p - 1) // 2 + 1)])
top = [(int(l), float(spec[l] / spec[1:].sum())) for l in np.argsort(-spec[1:])[:12] + 1]
out = dict(key_freqs=key, n_extra=n_extra, structured_set=sorted(int(x) for x in struct), key_only=acc_for(set(key)),
           structured_acc=acc_for(struct), random_mean=float(np.mean(accs)), random_max=float(np.max(accs)), random_min=float(np.min(accs)),
           n_random_draws=len(accs), c_spectrum_top12_share_of_block_energy=top)
json.dump(out, open(f"{run}/trigfit_gap_control.json", "w"), indent=1)
print(json.dumps(out, indent=1))
