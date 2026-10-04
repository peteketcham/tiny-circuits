"""Train on (a+b) mod p and/or (a*b) mod p; the task is given by the final token. Resumable and deterministic.
  python -m tiny_circuits.train_multi --p 53 --tasks addmul --seed 0 --steps 6000 --out runs/mt_p53/addmul/s0
"""
import argparse, json, os, pickle, time
import jax, jax.numpy as jnp, numpy as np, optax
from .model import init_params, forward, loss_fn
from .multitask import make_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=53)
    ap.add_argument("--tasks", choices=["add", "mul", "addmul"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_frac", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_mlp", type=int, default=512)
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--ckpt_every", type=int, default=250)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_seconds", type=float, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if not a.resume:
        json.dump(vars(a), open(f"{a.out}/config.json", "w"), indent=1)
    tasks = ["add", "mul"] if a.tasks == "addmul" else [a.tasks]
    (xtr, ytr), test, train_by = make_data(a.p, tasks, a.train_frac, a.seed)
    xtr, ytr = jnp.asarray(xtr), jnp.asarray(ytr)
    test = {t: tuple(map(jnp.asarray, v)) for t, v in test.items()}; train_by = {t: tuple(map(jnp.asarray, v)) for t, v in train_by.items()}
    params = init_params(jax.random.PRNGKey(a.seed), a.p, a.d_model, a.n_heads, a.d_mlp, n_tok=a.p + 2)
    opt = optax.adamw(a.lr, b1=0.9, b2=0.98, weight_decay=a.wd); state = opt.init(params)

    @jax.jit
    def step(params, state):
        loss, g = jax.value_and_grad(loss_fn)(params, xtr, ytr)
        upd, state = opt.update(g, state, params)
        return optax.apply_updates(params, upd), state, loss

    @jax.jit
    def evaluate(params):
        out = {}
        for t in tasks:
            xt, yt = test[t]; xr, yr = train_by[t]
            out[f"{t}_test_loss"] = loss_fn(params, xt, yt)
            out[f"{t}_test_acc"] = (forward(params, xt).argmax(-1) == yt).mean()
            out[f"{t}_train_acc"] = (forward(params, xr).argmax(-1) == yr).mean()
        return out

    log, start = [], 0
    if a.resume and os.path.exists(f"{a.out}/state.pkl"):
        st = pickle.load(open(f"{a.out}/state.pkl", "rb"))
        params = jax.tree_util.tree_map(jnp.asarray, st["params"]); state = jax.tree_util.tree_map(jnp.asarray, st["opt"])
        start = st["step"]; log = [r for r in json.load(open(f"{a.out}/log.json")) if r["step"] <= start]
        print(f"resumed at step {start}", flush=True)
    t0 = time.time() - (log[-1]["t"] if log else 0); t_session = time.time()
    for i in range(start, a.steps + 1):
        if a.max_seconds and time.time() - t_session > a.max_seconds and i % a.ckpt_every == 0:
            break
        if i % a.log_every == 0:
            r = {k: float(v) for k, v in evaluate(params).items()}
            r.update(step=i, train_loss=float(loss_fn(params, xtr, ytr)), t=time.time() - t0,
                     wnorm=float(jnp.sqrt(sum((v ** 2).sum() for v in params.values()))))
            log.append(r); print(json.dumps(r), flush=True); json.dump(log, open(f"{a.out}/log.json", "w"))
        if i % a.ckpt_every == 0:
            pickle.dump(dict(step=i, params=jax.tree_util.tree_map(np.asarray, params), opt=jax.tree_util.tree_map(np.asarray, state)),
                        open(f"{a.out}/state.pkl", "wb"))
        if i < a.steps:
            params, state, _ = step(params, state)
    if i >= a.steps:
        np.savez(f"{a.out}/final.npz", **{k: np.asarray(v) for k, v in params.items()})


if __name__ == "__main__":
    main()
