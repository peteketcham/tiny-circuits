import numpy as np


def modular_addition(p, train_frac, seed):
    """All (a,b) pairs for (a+b) mod p, split train/test by a seeded permutation."""
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    a, b = a.ravel(), b.ravel()
    tokens = np.stack([a, b, np.full_like(a, p)], axis=1).astype(np.int32)  # token p is '='
    labels = ((a + b) % p).astype(np.int32)
    perm = np.random.RandomState(seed).permutation(len(a))
    n_train = int(train_frac * len(a))
    tr, te = perm[:n_train], perm[n_train:]
    return (tokens[tr], labels[tr]), (tokens[te], labels[te])
