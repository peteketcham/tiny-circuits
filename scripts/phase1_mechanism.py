"""Phase 1 mechanism checks beyond ablation.
  (1) embedding / unembedding use the same key frequencies;
  (2) logits fit  sum_k a_k cos(w_k(a+b-c)) + b_k sin(w_k(a+b-c)) + class bias  (R^2);
  (3) activation patching: patch clean activations into a corrupted run, measure recovery of the clean answer.
  python scripts/phase1_mechanism.py runs/p113_s0"""
import sys, json
import numpy as np, jax.numpy as jnp
from tiny_circuits import analysis as A
from tiny_circuits.model import forward

run = sys.argv[1]; cfg = json.load(open(f"{run}/config.json")); p = cfg["p"]
P = A.load(f"{run}/final.npz"); keyf = [int(k) for k in A.key_frequencies(P, p)]
F, _ = A.fourier_basis(p); fr = A.freq_of_index(p); nf = (p - 1) // 2
out = {"key_freqs": keyf}

# (1) frequency content of W_E (input side) and W_U (output side)
def fnorms(M):  # M [p, d] -> norm per frequency
    pr = F @ np.asarray(M); return np.array([np.sqrt((pr[fr == k] ** 2).sum()) for k in range(nf + 1)])
nE, nU = fnorms(np.asarray(P["W_E"])[:p]), fnorms(np.asarray(P["W_U"]).T)
top = lambda n: sorted(int(k) for k in np.argsort(-n[1:])[:len(keyf)] + 1)
out["W_E_top_freqs"], out["W_U_top_freqs"] = top(nE), top(nU)
out["W_U_norm_share_in_key"] = float((nU[keyf] ** 2).sum() / (nU[1:] ** 2).sum())

# (2) trig-identity fit of the logits
rng = np.random.RandomState(0); n_pairs = 2000
a = rng.randint(0, p, n_pairs); b = rng.randint(0, p, n_pairs); c = np.arange(p)
L = np.asarray(A.full_logits(P, p))[a, b, :]                     # [n, p]
Y = (L - L.mean(1, keepdims=True)).ravel()
theta = (a + b)[:, None] - c[None, :]
cols = []
for k in keyf:
    w = 2 * np.pi * k / p
    cols += [np.cos(w * theta).ravel(), np.sin(w * theta).ravel()]
onehot = np.tile(np.eye(p) - 1.0 / p, (n_pairs, 1))              # class bias, centred
X = np.column_stack(cols + [onehot[:, :p - 1]])
coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
r2 = 1 - ((Y - X @ coef) ** 2).sum() / (Y ** 2).sum()
Xk = np.column_stack(cols); ck, *_ = np.linalg.lstsq(Xk, Y, rcond=None)
out["trig_fit_R2_with_class_bias"] = float(r2)
out["trig_fit_R2_trig_terms_only"] = float(1 - ((Y - Xk @ ck) ** 2).sum() / (Y ** 2).sum())
out["trig_coef_amplitudes"] = {k: float(np.hypot(coef[2 * i], coef[2 * i + 1])) for i, k in enumerate(keyf)}
# does the fit actually classify? argmax of the fitted logits vs truth
fit_logits = (X @ coef).reshape(n_pairs, p)
out["trig_fit_argmax_accuracy"] = float((fit_logits.argmax(1) == (a + b) % p).mean())

# (1b) what frequencies does each head's OV circuit (token -> residual) carry?
ov = {}
for h in range(P["W_Q"].shape[0]):
    n = fnorms(np.asarray(P["W_E"])[:p] @ np.asarray(P["W_V"][h]) @ np.asarray(P["W_O"][h]))
    ov[h] = dict(share_in_key=float((n[keyf] ** 2).sum() / (n[1:] ** 2).sum()),
                 per_key_freq_share={k: float(n[k] ** 2 / (n[keyf] ** 2).sum()) for k in keyf})
out["head_OV_frequency_content"] = ov

# (3) activation patching: clean (a,b) -> corrupted (a2,b2); how often does the *clean* answer come back?
N = 3000
a, b = rng.randint(0, p, N), rng.randint(0, p, N)
a2, b2 = (a + rng.randint(1, p, N)) % p, (b + rng.randint(1, p, N)) % p
mk = lambda x, y: jnp.asarray(np.stack([x, y, np.full_like(x, p)], 1).astype(np.int32))
clean, corr = mk(a, b), mk(a2, b2)
ans = (a + b) % p
_, cc = forward(P, clean, return_cache=True)
def acc(logits): return float((logits.argmax(-1) == ans).mean())
res = {"corrupted_baseline": acc(forward(P, corr)), "clean_baseline": acc(forward(P, clean))}
H = P["W_Q"].shape[0]
for h in range(H):  # patch one head's clean output into the corrupted run
    clean_ho = jnp.einsum("bhqe,hed->bhqd", cc["z"], P["W_O"])
    res[f"patch_head_{h}"] = acc(forward(P, corr, patch={"head_out": lambda x, h=h: x.at[:, h].set(clean_ho[:, h])}))
clean_ho = jnp.einsum("bhqe,hed->bhqd", cc["z"], P["W_O"])
res["patch_all_heads"] = acc(forward(P, corr, patch={"head_out": lambda x: clean_ho}))
res["patch_all_heads_but_h"] = {h: acc(forward(P, corr, patch={"head_out": lambda x, h=h: clean_ho.at[:, h].set(x[:, h])})) for h in range(H)}
res["patch_mlp_post"] = acc(forward(P, corr, patch={"mlp_post": lambda x: cc["mlp_post"]}))
dom, _ = A.neuron_frequency_profile(P, p)
for k in keyf:  # patch only the neurons whose dominant frequency is k
    m = jnp.asarray((dom == k).astype(np.float32))
    res[f"patch_neurons_freq{k}"] = acc(forward(P, corr, patch={"mlp_post": lambda x, m=m: x * (1 - m) + cc["mlp_post"] * m}))
out["activation_patching_clean_answer_accuracy"] = res
json.dump(out, open(f"{run}/mechanism.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
