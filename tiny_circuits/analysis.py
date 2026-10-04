"""Interpretability toolkit: Fourier basis, key-frequency finding, restricted/excluded loss,
ablations, and per-neuron frequency analysis. All functions take a params dict (numpy or jax)."""
import numpy as np
import jax, jax.numpy as jnp
from .model import forward
from .data import modular_addition


def fourier_basis(p):
    """Orthonormal real Fourier basis, rows = [const, cos1, sin1, cos2, sin2, ...] (p odd)."""
    x = np.arange(p)
    rows, names = [np.ones(p) / np.sqrt(p)], ["const"]
    for k in range(1, (p - 1) // 2 + 1):
        rows.append(np.cos(2 * np.pi * k * x / p) * np.sqrt(2 / p)); names.append(f"cos{k}")
        rows.append(np.sin(2 * np.pi * k * x / p) * np.sqrt(2 / p)); names.append(f"sin{k}")
    return np.stack(rows), names


def freq_of_index(p):
    """Frequency (0 for const, k for cos/sin k) of each Fourier basis row."""
    return np.array([0] + [k for k in range(1, (p - 1) // 2 + 1) for _ in (0, 1)])


def load(path):
    d = np.load(path)
    return {k: jnp.asarray(d[k]) for k in d.files}


def all_pairs(p):
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    a, b = a.ravel(), b.ravel()
    return jnp.asarray(np.stack([a, b, np.full_like(a, p)], 1).astype(np.int32)), (a + b) % p


def full_logits(params, p, patch=None):
    toks, _ = all_pairs(p)
    return forward(params, toks, patch=patch).reshape(p, p, p)  # [a,b,c]


def ce(logits_flat, labels):
    lp = jax.nn.log_softmax(logits_flat, -1)
    return float(-jnp.take_along_axis(lp, jnp.asarray(labels)[:, None], -1).mean())


def embed_fourier_norms(params, p):
    """Norm of W_E[:p] projected on each Fourier component, per frequency (sqrt of sum of squares)."""
    F, _ = fourier_basis(p)
    proj = F @ np.asarray(params["W_E"])[:p]            # [p, d]
    fr = freq_of_index(p)
    return np.array([np.sqrt((proj[fr == k] ** 2).sum()) for k in range((p - 1) // 2 + 1)])


def key_frequencies(params, p, min_frac=0.01):
    """Key frequencies = those k whose (k,k) 2D-Fourier pair holds >= min_frac of the logit
    energy that varies with the output class (const-in-(a,b) component excluded from the
    denominator). Cross-check against embed_fourier_norms."""
    F, _ = fourier_basis(p)
    L = np.asarray(full_logits(params, p))
    Lf = np.einsum("ia,jb,abc->ijc", F, F, L, optimize=True)
    E = ((Lf - Lf.mean(-1, keepdims=True)) ** 2).sum(-1)
    fr = freq_of_index(p)
    pair = np.array([E[np.ix_(fr == k, fr == k)].sum() for k in range((p - 1) // 2 + 1)])
    return np.where(pair[1:] / pair[1:].sum() >= min_frac)[0] + 1


def _component_mask(p, freqs):
    """Mask over the 2D (a,b) Fourier grid keeping components where both freqs are in the key set."""
    fr = freq_of_index(p)
    k = np.isin(fr, freqs)
    return k[:, None] & k[None, :]


def restricted_excluded_loss(params, p, freqs, split=None):
    """Progress measures after Nanda et al. Logits [a,b,c] -> 2D Fourier over (a,b).
    key   = components with both a- and b-frequency in `freqs`
    const = the (0,0) component (a class-dependent bias; large in practice, must be kept)
    restricted = key + const ;  excluded = full - key.
    Returns CE of (full, restricted, excluded) on `split` = (tokens, labels), or all pairs."""
    F, _ = fourier_basis(p)
    L = np.asarray(full_logits(params, p))
    Lf = np.einsum("ia,jb,abc->ijc", F, F, L, optimize=True)
    rec = lambda x: np.einsum("ia,jb,ijc->abc", F, F, x, optimize=True)
    mk = _component_mask(p, freqs)[:, :, None]
    mc = np.zeros((p, p, 1), bool); mc[0, 0] = True
    key, const = rec(Lf * mk), rec(Lf * mc)
    restricted, excluded = key + const, L - key
    if split is None:
        toks, labels = all_pairs(p); idx = np.arange(p * p)
    else:
        toks, labels = split; idx = np.asarray(toks)[:, 0] * p + np.asarray(toks)[:, 1]
    pick = lambda x: x.reshape(p * p, p)[idx]
    return tuple(ce(jnp.asarray(pick(x)), labels) for x in (L, restricted, excluded))


def head_ablation(params, p, split, mode="zero"):
    """CE loss on `split` after ablating each attention head (zero its z)."""
    toks, labels = split
    out = []
    for h in range(params["W_Q"].shape[0]):
        def f(z, h=h): return z
        # ablate by zeroing the head's pattern-weighted values via the pattern patch: set pattern row to 0
        def pat(pattern, h=h): return pattern.at[:, h].set(0.0)
        out.append(ce(forward(params, jnp.asarray(toks), patch={"pattern": pat}), labels))
    return out


def neuron_ablation_curve(params, p, split, order):
    """Loss as neurons are zeroed cumulatively in `order` (list of neuron indices)."""
    toks, labels = split
    losses = []
    for n in [0, 8, 16, 32, 64, 128, 256, 512]:
        mask = np.ones(params["W_in"].shape[1], dtype=np.float32); mask[order[:n]] = 0
        losses.append((n, ce(forward(params, jnp.asarray(toks), patch={"mlp_post": lambda x: x * mask}), labels)))
    return losses


def neuron_frequency_profile(params, p):
    """For each MLP neuron, fraction of its [a,b]-activation variance in each frequency pair (a-axis).
    Returns dominant freq per neuron and the explained fraction."""
    toks, _ = all_pairs(p)
    _, cache = forward(params, toks, return_cache=True)
    act = np.asarray(cache["mlp_post"][:, -1, :]).reshape(p, p, -1)      # [a,b,n]
    F, _ = fourier_basis(p)
    Af = np.einsum("ia,jb,abn->ijn", F, F, act, optimize=True)
    fr = freq_of_index(p)
    nf = (p - 1) // 2
    # energy per frequency (a-axis freq k, b-axis freq k), ignoring const
    energy = np.stack([(Af[np.ix_(fr == k, fr == k)] ** 2).sum((0, 1)) for k in range(1, nf + 1)])  # [k,n]
    total = (Af ** 2).sum((0, 1)) - Af[0, 0] ** 2 + 1e-12
    dom = energy.argmax(0) + 1
    return dom, energy.max(0) / total
