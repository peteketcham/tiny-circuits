"""Multi-task data and generic cyclic-group Fourier analysis.

Tasks: 'add' = (a+b) mod p  (final token p),  'mul' = (a*b) mod p  (final token p+1).
Multiplication of nonzero residues is cyclic addition of discrete logs (period p-1), so the same
Fourier analysis applies in log coordinates (n = p-1) after dropping the value 0.
"""
import numpy as np

TASK_ID = {"add": 0, "mul": 1}


def primitive_root(p):
    for g in range(2, p):
        x, seen = 1, set()
        for _ in range(p - 1):
            x = x * g % p; seen.add(x)
        if len(seen) == p - 1:
            return g
    raise ValueError("no primitive root (p not prime?)")


def log_tables(p):
    g = primitive_root(p)
    exp = np.array([pow(g, i, p) for i in range(p - 1)])
    log = -np.ones(p, dtype=int); log[exp] = np.arange(p - 1)
    return g, exp, log


def task_pairs(p, task):
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij"); a, b = a.ravel(), b.ravel()
    c = (a + b) % p if task == "add" else (a * b) % p
    tok = np.stack([a, b, np.full_like(a, p + TASK_ID[task])], 1).astype(np.int32)
    return tok, c.astype(np.int32)


def make_data(p, tasks, train_frac, seed):
    """Per-task random train/test split. Returns (x_train, y_train), {task: (x_test, y_test)}, {task: (x_train, y_train)}."""
    xs, ys, test, train_by = [], [], {}, {}
    for t in tasks:
        tok, lab = task_pairs(p, t)
        perm = np.random.RandomState(seed * 10 + TASK_ID[t]).permutation(len(lab)); n = int(train_frac * len(lab))
        tr, te = perm[:n], perm[n:]
        xs.append(tok[tr]); ys.append(lab[tr]); test[t] = (tok[te], lab[te]); train_by[t] = (tok[tr], lab[tr])
    return (np.concatenate(xs), np.concatenate(ys)), test, train_by


def fourier_basis_n(n):
    """Orthonormal real Fourier basis of Z_n. Returns (F [n,n], freq [n]); even n gets a single Nyquist row."""
    x = np.arange(n); rows, fr = [np.ones(n) / np.sqrt(n)], [0]
    for k in range(1, (n - 1) // 2 + 1):
        rows.append(np.cos(2 * np.pi * k * x / n) * np.sqrt(2 / n)); fr.append(k)
        rows.append(np.sin(2 * np.pi * k * x / n) * np.sqrt(2 / n)); fr.append(k)
    if n % 2 == 0:
        rows.append(((-1.0) ** x) / np.sqrt(n)); fr.append(n // 2)
    return np.stack(rows), np.array(fr)


def _lf(L, F):
    return np.einsum("ia,jb,abc->ijc", F, F, L, optimize=True)


def key_freqs_generic(L, n, min_frac=0.01):
    """L: [n,n,n] logits indexed (a,b,class), class order = cyclic order. Same rule as analysis.key_frequencies."""
    F, fr = fourier_basis_n(n); Lf = _lf(L, F)
    E = ((Lf - Lf.mean(-1, keepdims=True)) ** 2).sum(-1)
    ks = sorted(set(int(k) for k in fr if k > 0))
    pair = np.array([E[np.ix_(fr == k, fr == k)].sum() for k in ks])
    return [k for k, v in zip(ks, pair / pair.sum()) if v >= min_frac]


def _ce(logits, labels):
    z = logits - logits.max(-1, keepdims=True)
    lp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    return float(-lp[np.arange(len(labels)), labels].mean())


def restricted_excluded_generic(L, n, freqs, ia, ib, labels):
    """Same definition as analysis.restricted_excluded_loss, for any n. (ia, ib) index the evaluated pairs."""
    F, fr = fourier_basis_n(n); Lf = _lf(L, F)
    rec = lambda x: np.einsum("ia,jb,ijc->abc", F, F, x, optimize=True)
    k = np.isin(fr, freqs); mk = (k[:, None] & k[None, :])[:, :, None]
    mc = np.zeros((n, n, 1), bool); mc[0, 0] = True
    key, const = rec(Lf * mk), rec(Lf * mc)
    pick = lambda X: X[ia, ib, :]
    return tuple(_ce(pick(X), labels) for X in (L, key + const, L - key))
