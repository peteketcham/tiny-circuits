"""Score the pre-registered predictions (PLAN.md Q1, Q2, Q4) against a sweep summary, with null models.
  python scripts/phase2_compare.py runs/sweep_p53 [tag ...]     -> prints and writes <dir>/compare_<tag>.json"""
import sys, json, itertools
import numpy as np

d = sys.argv[1]; tags = sys.argv[2:] or ["main"]
for tag in tags:
    rows = json.load(open(f"{d}/summary_{tag}.json")); p = 53; K = (p - 1) // 2
    g = [r for r in rows if r["status"] == "grokked"]; out = {"tag": tag, "n_seeds": len(rows), "n_grokked": len(g)}
    # Q1
    sizes = [len(r["key_freqs"]) for r in g]
    out["Q1"] = dict(grok_rate=len(g) / len(rows), size_min=min(sizes), size_max=max(sizes),
                     size_counts={int(s): sizes.count(s) for s in sorted(set(sizes))},
                     prediction_holds=bool(len(g) / len(rows) >= 0.9 and min(sizes) >= 2 and max(sizes) <= 7))
    # Q2
    sets = [frozenset(r["key_freqs"]) for r in g]
    cnt = np.zeros(K + 1, int)
    for s in sets:
        for k in s: cnt[k] += 1
    def stats(ss):
        c = np.zeros(K + 1, int)
        for s in ss:
            for k in s: c[k] += 1
        jac = [len(a & b) / len(a | b) for a, b in itertools.combinations(ss, 2)]
        return c[1:].max(), float(np.mean(jac)), sum(1 for a, b in itertools.combinations(ss, 2) if a == b)
    mx, mj, ident = stats(sets)
    rng = np.random.RandomState(0); null = [stats([frozenset(rng.choice(np.arange(1, K + 1), len(s), replace=False)) for s in sets]) for _ in range(5000)]
    nmx, nmj = np.array([x[0] for x in null]), np.array([x[1] for x in null])
    order = np.argsort(-cnt[1:])[:8] + 1
    out["Q2"] = dict(appearance_counts_top={int(k): int(cnt[k]) for k in order}, max_appearance=int(mx), max_appearance_frac=float(mx / len(g)),
                     null_max_appearance_mean=float(nmx.mean()), null_max_appearance_p95=float(np.percentile(nmx, 95)),
                     p_value_max_appearance=float((nmx >= mx).mean()),
                     mean_pairwise_jaccard=mj, null_mean_jaccard=float(nmj.mean()), p_value_jaccard=float((nmj >= mj).mean()),
                     identical_pairs=int(ident), prediction_holds=bool(mx / len(g) <= 0.5))
    # Q4
    strict, loose = 0, 0
    for r in g:
        hs = r["heads_dominant_freq_share"]
        byf = {}
        for f, sh in hs: byf.setdefault(f, []).append(sh)
        loose += any(len(v) >= 2 for v in byf.values())
        strict += any(sum(1 for s in v if s >= 0.5) >= 2 for v in byf.values())
    out["Q4"] = dict(strict_rule_seeds=strict, loose_same_dominant_freq_seeds_POSTHOC=loose, of=len(g),
                     prediction_holds=bool(strict / len(g) > 0.5))
    # Circuit validity across all grokked seeds
    out["validity"] = dict(max_restricted_over_full=float(max(r["test_loss"]["restricted"] / max(r["test_loss"]["full"], 1e-12) for r in g)),
                           worst_restricted_loss=float(max(r["test_loss"]["restricted"] for r in g)),
                           min_excluded_loss=float(min(r["test_loss"]["excluded"] for r in g)),
                           min_control_restricted=float(min(r["control"]["restricted_min"] for r in g)),
                           max_control_excluded=float(max(r["control"]["excluded_max"] for r in g)))
    gs = [r["grok_step_50"] for r in g]; out["grok_step_50"] = dict(median=float(np.median(gs)), min=int(min(gs)), max=int(max(gs)))
    json.dump(out, open(f"{d}/compare_{tag}.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
