"""Post hoc control: analyse single-task *multiplication* models in the plain Z_p basis (wrong coordinates) instead of discrete-log coordinates.
If the log-space reading is the right one, the key-frequency set found here should be large and its restricted loss poor.
  PYTHONPATH=. python scripts/multitask_mul_additive_control.py runs/mt_p53/mul -> <dir>/mul_additive_control.json"""
import sys, glob, json, numpy as np, jax.numpy as jnp
from tiny_circuits import analysis as A, multitask as M
from tiny_circuits.model import forward
root = sys.argv[1]; out = []
for d in sorted(glob.glob(f"{root}/s[0-9]*")):
    cfg = json.load(open(f"{d}/config.json")); p = cfg["p"]; P = A.load(f"{d}/final.npz")
    tok, lab = M.task_pairs(p, "mul")
    L = np.asarray(forward(P, jnp.asarray(tok)), dtype=np.float64).reshape(p, p, p)
    _, test, _ = M.make_data(p, ["mul"], cfg["train_frac"], cfg["seed"]); xt, yt = test["mul"]
    key = M.key_freqs_generic(L, p)
    full, restr, excl = M.restricted_excluded_generic(L, p, key, xt[:, 0], xt[:, 1], yt)
    out.append(dict(seed=cfg["seed"], n_key_additive_basis=len(key), full=full, restricted=restr, excluded=excl))
    print(out[-1])
json.dump(out, open(f"{root}/mul_additive_control.json", "w"), indent=1)
