"""Score PLAN.md Addendum A (Q5a-d): composite p=45 vs prime p=47.
  python scripts/phase2_q5.py runs/sweep_p45 runs/sweep_p47  -> runs/q5.json"""
import sys, json
from math import gcd
import numpy as np

rng = np.random.RandomState(0)
A = [r for r in json.load(open(f"{sys.argv[1]}/summary_main.json"))]
B = [r for r in json.load(open(f"{sys.argv[2]}/summary_main.json"))]
gA = [r for r in A if r["status"] == "grokked"]; gB = [r for r in B if r["status"] == "grokked"]
S = {k for k in range(1, 23) if gcd(k, 45) > 1}                       # 10 of the 22 frequencies at p=45
out = dict(S=sorted(S))

def perm_count(rows, K, reps=5000):
    obs = sum(len(S & set(r["key_freqs"])) for r in rows); tot = sum(len(r["key_freqs"]) for r in rows)
    null = np.array([sum(len(S & set(rng.choice(np.arange(1, K + 1), len(r["key_freqs"]), replace=False))) for r in rows) for _ in range(reps)])
    p2 = float(min(1.0, 2 * min((null >= obs).mean(), (null <= obs).mean())))
    return dict(observed=int(obs), total_key_freqs=int(tot), observed_frac=obs / tot, null_mean_frac=float(null.mean() / tot), p_two_sided=p2)

sz = lambda rows: [len(r["key_freqs"]) for r in rows]
out["Q5a"] = {name: dict(grokked=len(g), of=len(rows), grok_rate=len(g) / len(rows), size_min=min(sz(g)), size_max=max(sz(g)),
                         size_counts={int(s): sz(g).count(s) for s in sorted(set(sz(g)))}, failed_seeds=[(r["seed"], r["final_test_acc"]) for r in rows if r["status"] != "grokked"])
              for name, rows, g in (("p45", A, gA), ("p47", B, gB))}
out["Q5a"]["prediction_holds"] = bool(all(v["grok_rate"] >= 0.9 and 2 <= v["size_min"] and v["size_max"] <= 7 for v in (out["Q5a"]["p45"], out["Q5a"]["p47"])))
out["Q5b_p45_gcd_enrichment"] = perm_count(gA, 22); out["Q5b_p45_gcd_enrichment"]["prediction_holds"] = bool(out["Q5b_p45_gcd_enrichment"]["p_two_sided"] > 0.05)
out["Q5c_p47_placebo"] = perm_count(gB, 23); out["Q5c_p47_placebo"]["prediction_holds"] = bool(out["Q5c_p47_placebo"]["p_two_sided"] > 0.05)
a, b = np.array(sz(gA)), np.array(sz(gB)); d = a.mean() - b.mean(); pool = np.concatenate([a, b])
nd = []
for _ in range(5000):
    rng.shuffle(pool); nd.append(pool[:len(a)].mean() - pool[len(a):].mean())
out["Q5d_mean_size"] = dict(p45=float(a.mean()), p47=float(b.mean()), diff=float(d), p_two_sided=float((np.abs(nd) >= abs(d)).mean()),
                            prediction_holds=bool(abs(d) < 1 and (np.abs(nd) >= abs(d)).mean() > 0.05))
# descriptive (not pre-registered)
from collections import Counter
for name, g in (("p45", gA), ("p47", gB)):
    c = Counter(k for r in g for k in r["key_freqs"]); out[f"DESCRIPTIVE_{name}_most_common"] = c.most_common(8)
    out[f"DESCRIPTIVE_{name}_grok_step_median"] = float(np.median([r["grok_step_50"] for r in g]))
    out[f"DESCRIPTIVE_{name}_validity"] = dict(max_restricted_over_full=float(max(r["test_loss"]["restricted"] / max(r["test_loss"]["full"], 1e-12) for r in g)),
                                              min_excluded=float(min(r["test_loss"]["excluded"] for r in g)),
                                              min_control_restricted=float(min(r["control"]["restricted_min"] for r in g)))
json.dump(out, open("runs/q5.json", "w"), indent=1)
print(json.dumps(out, indent=1))
