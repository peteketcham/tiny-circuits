"""Train the tiny transformer on (a+b) mod p. One command, fully seeded.

  python -m tiny_circuits.train --p 113 --seed 0 --steps 10000 --out runs/p113_s0
"""
import argparse, json, os, time
import jax, jax.numpy as jnp, numpy as np, optax
from .model import init_params, forward, loss_fn
from .data import modular_addition


def accuracy(params, tokens, labels):
    return (forward(params, tokens).argmax(-1) == labels).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=113)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_seed", type=int, default=None)
    ap.add_argument("--train_frac", type=float, default=0.3)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_mlp", type=int, default=512)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(f"{a.out}/config.json", "w"), indent=1)

    (xtr, ytr), (xte, yte) = modular_addition(a.p, a.train_frac,
                                              a.seed if a.data_seed is None else a.data_seed)
    xtr, ytr, xte, yte = map(jnp.asarray, (xtr, ytr, xte, yte))
    params = init_params(jax.random.PRNGKey(a.seed), a.p, a.d_model, a.n_heads, a.d_mlp)
    opt = optax.adamw(a.lr, b1=0.9, b2=0.98, weight_decay=a.wd)
    state = opt.init(params)

    @jax.jit
    def step(params, state):
        loss, g = jax.value_and_grad(loss_fn)(params, xtr, ytr)
        upd, state = opt.update(g, state, params)
        return optax.apply_updates(params, upd), state, loss

    @jax.jit
    def evaluate(params):
        return (loss_fn(params, xte, yte), accuracy(params, xtr, ytr), accuracy(params, xte, yte))

    log, t0 = [], time.time()
    for i in range(a.steps + 1):
        if i % a.log_every == 0:
            tl, tra, tea = map(float, evaluate(params))
            trl = float(loss_fn(params, xtr, ytr))
            log.append(dict(step=i, train_loss=trl, test_loss=tl, train_acc=tra, test_acc=tea,
                            wnorm=float(jnp.sqrt(sum((v ** 2).sum() for v in params.values()))),
                            t=time.time() - t0))
            print(json.dumps(log[-1]), flush=True)
            json.dump(log, open(f"{a.out}/log.json", "w"))
        if i % a.ckpt_every == 0:
            np.savez(f"{a.out}/ckpt_{i:06d}.npz", **{k: np.asarray(v) for k, v in params.items()})
        if i < a.steps:
            params, state, _ = step(params, state)
    np.savez(f"{a.out}/final.npz", **{k: np.asarray(v) for k, v in params.items()})


if __name__ == "__main__":
    main()
