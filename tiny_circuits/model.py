"""Minimal 1-layer transformer in plain JAX (no LayerNorm, learned pos-emb, ReLU MLP).

Mirrors the setup used in the grokking / mechanistic-interpretability literature so
Phase 1 results are directly comparable to published ones.
Inputs are [a, b, '='] token triples; the output is read off the last position.
"""
import jax
import jax.numpy as jnp
import numpy as np


def init_params(key, p, d_model=128, n_heads=4, d_mlp=512, n_ctx=3):
    d_head = d_model // n_heads
    ks = jax.random.split(key, 8)
    s = lambda fan_in: 1.0 / np.sqrt(fan_in)
    return {
        "W_E": jax.random.normal(ks[0], (p + 1, d_model)) * s(d_model),
        "W_pos": jax.random.normal(ks[1], (n_ctx, d_model)) * s(d_model),
        "W_Q": jax.random.normal(ks[2], (n_heads, d_model, d_head)) * s(d_model),
        "W_K": jax.random.normal(ks[3], (n_heads, d_model, d_head)) * s(d_model),
        "W_V": jax.random.normal(ks[4], (n_heads, d_model, d_head)) * s(d_model),
        "W_O": jax.random.normal(ks[5], (n_heads, d_head, d_model)) * s(d_model),
        "W_in": jax.random.normal(ks[6], (d_model, d_mlp)) * s(d_model),
        "b_in": jnp.zeros((d_mlp,)),
        "W_out": jax.random.normal(ks[7], (d_mlp, d_model)) * s(d_mlp),
        "b_out": jnp.zeros((d_model,)),
        "W_U": jax.random.normal(jax.random.fold_in(key, 99), (d_model, p)) * s(d_model),
    }


def forward(params, tokens, return_cache=False, patch=None):
    """tokens: int array [batch, 3]. Returns logits [batch, p] at the final position.

    patch: optional dict of callables applied to named activations, e.g.
      {"attn_out": f, "mlp_post": f, "resid_mid": f, "pattern": f}; f(x) -> x'.
    """
    patch = patch or {}
    x = params["W_E"][tokens] + params["W_pos"][None, :, :]           # [B,T,D]
    q = jnp.einsum("btd,hde->bhte", x, params["W_Q"])
    k = jnp.einsum("btd,hde->bhte", x, params["W_K"])
    v = jnp.einsum("btd,hde->bhte", x, params["W_V"])
    scores = jnp.einsum("bhqe,bhke->bhqk", q, k) / jnp.sqrt(q.shape[-1])
    T = tokens.shape[1]
    mask = jnp.tril(jnp.ones((T, T), dtype=bool))
    scores = jnp.where(mask[None, None], scores, -1e9)
    pattern = jax.nn.softmax(scores, axis=-1)
    if "pattern" in patch:
        pattern = patch["pattern"](pattern)
    z = jnp.einsum("bhqk,bhke->bhqe", pattern, v)
    attn_out = jnp.einsum("bhqe,hed->bqd", z, params["W_O"])
    if "attn_out" in patch:
        attn_out = patch["attn_out"](attn_out)
    resid_mid = x + attn_out
    if "resid_mid" in patch:
        resid_mid = patch["resid_mid"](resid_mid)
    pre = resid_mid @ params["W_in"] + params["b_in"]
    post = jax.nn.relu(pre)
    if "mlp_post" in patch:
        post = patch["mlp_post"](post)
    mlp_out = post @ params["W_out"] + params["b_out"]
    resid_post = resid_mid + mlp_out
    logits = resid_post[:, -1, :] @ params["W_U"]
    if return_cache:
        return logits, dict(pattern=pattern, attn_out=attn_out, resid_mid=resid_mid,
                            mlp_pre=pre, mlp_post=post, mlp_out=mlp_out,
                            resid_post=resid_post, v=v, z=z)
    return logits


def loss_fn(params, tokens, labels, patch=None):
    logits = forward(params, tokens, patch=patch)
    logp = jax.nn.log_softmax(logits.astype(jnp.float64) if jax.config.jax_enable_x64 else logits)
    nll = -jnp.take_along_axis(logp, labels[:, None], axis=-1)[:, 0]
    return nll.mean()
