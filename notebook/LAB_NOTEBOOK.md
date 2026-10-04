# Lab notebook

Newest entries last. Failures and mistakes are recorded on purpose.

## 2026-10-03 — Phase 1 setup and first run

**Environment.** Cloud sandbox, 2 CPU cores, 7 GB RAM, Python 3.13. The PyTorch CPU wheel host
(`download.pytorch.org`) is blocked by the sandbox proxy, so the project uses JAX (`jax[cpu]` 0.11.2 from PyPI).

**What went wrong, in order**

1. *Background training died, twice.* I launched training with `nohup ... &` and it was killed once
   after step 4000, then again at step ~2200 after I added `setsid`. Background processes appear to be
   reaped when a turn in the session ends. Fix: `train.py` now saves params and optimizer state to
   `state.pkl` every `--ckpt_every` steps and supports `--resume` and `--max_seconds`, and I ran the
   job as five foreground chunks of ≤8 minutes. Lesson for later phases: **design every long job to be
   chunked and resumable from the start.**
2. *Resume exactness was checked, not assumed.* Training is full-batch with no RNG after init, so a resume with
   restored optimizer state should be exact. Check: the first (crashed) run logged train loss
   6.509822583211644e-07 and test loss 25.073820114135742 at step 4000; the resumed run, which crossed a
   chunk boundary at step 3400, logged identical values. Same at step 2200. So: bit-exact on this machine.
3. *Analysis script took >10 minutes.* `np.einsum("ia,jb,abc->ijc", ...)` without `optimize=True` evaluates
   a naive 5-index product (~1e10 operations per call). Fixed with `optimize=True`; the full analysis now
   runs in about a minute.
4. *First restricted-loss result was wrong, and I changed the analysis after seeing it.*
   First version: restricted loss 4.86 (chance), excluded 26.4. That is not what a clean Fourier circuit
   gives, so I debugged the toolkit rather than the claim. Two causes:
   - The restricted set left out the (0,0) Fourier component of the logits. That component is a large,
     class-dependent bias (23% of logit energy; logit range −195 to +64). Without it the key components
     cannot overcome it. Fix: restricted = key components + (0,0); excluded = full − key components.
   - Key frequencies were chosen as "embedding norm > 10% of max", which also picked up k=1
     (embedding norm 2.5, but ~0.1% of logit energy). Fix: key frequencies are now defined from the
     **logits** (frequencies whose (k,k) pair holds ≥1% of the class-varying energy), with the embedding
     top-k as a cross-check. They agree exactly: {24, 28, 46, 56}.
   Both changes were made after seeing the failing numbers. I added a control (20 random 4-frequency sets)
   so the final result does not depend on that choice: random sets give restricted loss ≥ 25.8.
   The unexplained part: why k=1 has an embedding norm of 2.5 and no logit presence. Not investigated.

**Result.** See README claims 1–10 (claims 7–10 come from the mechanism checks below). Grokking onset (test acc > 50%) at step 8600, 100% by step 10000.

**Open observations**
- Training loss has regular sharp spikes about every 1.4k steps, in both train and test curves. Looks like
  the Adam-related "slingshot" effect reported in the grokking literature, but I have not tested that.
- Neuron-level analysis is weak: median explained fraction 0.22, none > 0.9. The metric includes
  cross terms and ReLU harmonics; I do not yet have a better one.

### Mechanism checks (same day, later)

`scripts/phase1_mechanism.py`, results in `runs/p113_s0/mechanism.json`.
- `W_E` and `W_U` both have top-4 frequencies {24, 28, 46, 56}.
- Per-head OV content: heads 0 and 2 are nearly identical (k=46: 80.88% and 80.89% of key-frequency norm²);
  heads 1 and 3 likewise (k=24: 86.0% and 85.7%). Two duplicated pairs. Worth testing in Phase 2 whether
  the number of duplicates depends on seed or head count.
- Activation patching: no single head restores the clean answer; all heads together restore 100%; any three
  restore about a third. The answer is built from all four jointly.
- **The clean trig-identity form did not hold up.** Fitting logits to Σₖ aₖcos+bₖsin of wₖ(a+b−c) plus a
  class bias gives R² = 0.82 but only 66.7% argmax accuracy, so the textbook description is incomplete for
  this model. I expected it to fit better than that. Not yet explained; I would not write "the network
  implements the trig identity" on this evidence.
- `model.py` gained an optional `head_out` patch hook. The default path is untouched, and I re-checked that
  the final model's test loss is bit-identical (5.786600922874641e-06) after the edit.

**Next.** Understand the gap in the trig fit (amplitude dependence on (a,b)? cross-frequency terms?), then
seed universality at reduced cost: smaller p (e.g. 53) so many seeds fit in the compute budget.

### Sizing the seed sweep (exploratory, 2026-10-03 evening)

Goal: a cheaper setting than p=113 (~0.13 s/step, ~25 min per seed to grok) so many seeds fit in the budget.
- p=53, train_frac 0.3 (842 training pairs), 15,000 steps, ~0.02 s/step: **did not grok** (test acc 4.6% at the end).
  The training set is too small for this weight decay and step budget. (The log for that attempt was deleted
  and not committed; the numbers above are from the console output.)
- p=53, train_frac 0.5: interrupted by the user at step ~2250, but already at **94.7% test accuracy** (log in
  `runs/sweep_p53/s0/`). So this setting groks quickly (under ~2.5k steps, roughly 1 minute), which makes a
  many-seed sweep affordable. It was not run to completion, so final accuracy and circuit are unchecked.
  Note this is a different split than Phase 1 (50% vs 30%), so sweep results will not be directly comparable.

## 2026-10-03/04 — Phase 2 seed sweep (p=53, 50% train, 6000 steps). Plan: `PLAN.md`

Note: the exploratory `runs/sweep_p53/s0` files mentioned above were deleted from the working tree when the
real sweep began (same directory name); they remain in git history (commit before `db9cfa7`).

### C1 (8 seeds done, ~4.5 min/seed)
- 8/8 grokked. Stop condition (restricted loss >10× full loss) did not trigger. Random-frequency controls fail
  on every seed (restricted min 7–21 vs ≲2e-5 for the key set). Key-frequency counts: 3–5 (five seeds with 4, two with 3, one with 5; see `summary_c1.json`).
- Bug fixed in `phase2_analysis.py`: seed-directory glob matched `summary_*.json`; now `s[0-9]*`.
- **Q2 is under pressure.** Frequency 9 appears in 5/8 seeds' key sets, 7 in 4/8, 21 in 3/8. Pre-registered
  prediction was "no frequency in more than ~50% of seeds". Uniform null: a given frequency appears with
  probability ~4/26 ≈ 15% per seed. Will test properly (permutation null) at C2 instead of eyeballing. Possible
  reasons to check: some frequencies are easier to learn; split/seed correlations.
- **Q3 cannot be assessed as designed.** Grokking is fast (test acc >50% by step 500–1000 in most seeds, 2250 in the
  slowest), so for 5/8 seeds there is no checkpoint at ckpt_every=250 where test acc is still <20%
  (`early_J` = NaN). Training is bit-exact, so the fix is to re-run seeds with checkpoints every 25 steps for
  the first 1500 steps, keeping the pre-registered definition ("last checkpoint with test acc < 20%"). Do this
  at C2/C3.
- **Q4 definition is strict.** I wrote "share ≥ 50%". Head shares here are 0.34–0.76, and several seeds have
  two heads on the *same* dominant frequency with shares of ~0.45 (e.g. seed 1: freqs 9, 9, 17, 17). I will
  report Q4 by the pre-registered rule and separately report the looser "same dominant frequency" count,
  labelled as post hoc.

### C2 (all 30 seeds done, 00:35 CDT) — scoring PLAN.md Q1, Q2, Q4

Scripts: `phase2_analysis.py` → `runs/sweep_p53/summary_<tag>.json`; `phase2_compare.py` → `compare_<tag>.json`.
Tags: `main` (1% key-frequency threshold, 99% grok cutoff), `f005`/`f02` (0.5% / 2% threshold), `g98` (98% cutoff).

- **Q1 holds.** 29/30 seeds reach ≥99% test accuracy; key-frequency counts are 3 (6 seeds), 4 (18), 5 (5).
  The miss is seed 8, a slow grokker: 96% at step 3000, then 98.9–99.3% for the rest of training, so it fell
  0.1 point under my cutoff at step 6000. It stays in the tables as "no_grok" under the pre-registered rule.
  Sensitivity (`g98`, cutoff 98%): seed 8 has a valid 5-frequency circuit (restricted 1.1e-4 vs full 5.0e-2);
  nothing below changes.
- **Q2 holds, and the C1 worry was small-sample noise.** At 8 seeds frequency 9 looked over-represented. At 29 seeds the
  most common frequency (9) appears in 9 seeds (31%); a random-draw null with the same set sizes (5000 draws)
  gives a mean maximum of 8.6 (95th percentile 11), p = 0.50. Mean pairwise Jaccard between seeds' sets is 0.087 vs
  0.091 under the null (p = 0.80); zero pairs of seeds have identical sets. Results are the same at thresholds
  0.5% and 2% (p = 0.58/0.84 and 0.86/0.91). So: no evidence that seeds favour particular frequencies. This is
  absence of evidence at n≈29, not proof of uniformity.
- **Q4 fails as pre-registered.** Rule: ≥2 heads with the same dominant frequency, each with share ≥ 50% of
  key-frequency norm². Met in 13/29 seeds (45%); I predicted a majority.
  Post hoc, loosening to "≥2 heads share a dominant frequency" gives 28/29, **but that statistic is nearly
  uninformative**: with 4 heads and 3–5 key frequencies, random assignment already makes a repeat likely
  (certain for 3 frequencies; 91% for 4; 81% for 5; expected ≈ 26 of 29 seeds). The Phase 1 model's duplicated heads
  were remarkable because shares were ~80–86% and nearly identical between heads, not because two heads shared a label.
  I make no "duplicate heads" claim for the sweep.
- **Circuit validity across all grokked seeds:** worst restricted/full ratio 1.004 at the 1% threshold (1.8 at 2%,
  where a key frequency starts to be dropped); smallest excluded loss 8.2; random-frequency controls: restricted
  ≥ 7.1, excluded ≤ 0.013. Time to 50% test accuracy: median 750 steps (range 500–2250).
- **Not yet done:** Q3 (needs early checkpoints; below) and the re-training determinism check from PLAN.md C2.

### Q3 — early prediction of the key frequencies (`runs/sweep_p53/q3.json`, `scripts/phase2_q3.py`)

Regenerated each seed's first 1500 steps with a checkpoint every 25 steps (`runs/sweep_p53_early/`, not committed).
**Determinism check:** weights at step 1500 are bit-identical to the original sweep in 30/30 seeds, and all logged train/test
losses over steps 0–1500 match exactly.
- **Q3 holds.** At the last checkpoint with test accuracy < 20% (median: step 100), the top-n embedding
  frequencies match the final key set with mean Jaccard 0.75 (median 0.67; 86% of seeds above 0.5) against a chance
  level of 0.093. Predicted: > 0.5.
- Bug caught while scoring: the first version of the chance baseline drew two *different* random sets for the
  intersection and the union (gave 0.083). Fixed to draw once (0.093). The conclusion did not change.
- **Post hoc, not pre-registered:** mean Jaccard by step: 0 → 0.13, 50 → 0.55, 100 → 0.75, 200 → 0.76, 500 → 0.80,
  1000 → 0.83, 1500 → 0.93. So the frequencies are largely settled within the first ~100 steps, while test accuracy is
  still near chance and long before the generalisation jump (median step 750). It keeps creeping up afterwards, so some
  seeds still change a frequency later. Step 0 is 0.13 against a chance of 0.093; with 29 seeds I cannot tell
  whether that is initialisation bias or noise.
- Caveat: n (the number of frequencies) is taken from the final model, which makes the prediction a little easier than
  predicting n as well.

### Q5 — composite (p=45) vs prime (p=47), 24 seeds each (`runs/q5.json`, `scripts/phase2_q5.py`)

Pre-registered in PLAN.md Addendum A (committed before the runs). Deviation from the original Q5 (p=105) explained there.
- **Q5a holds.** p=45: 24/24 reach ≥99%; p=47: 23/24 (seed 19 reached 95.9% and is excluded under the same rule).
  Key-frequency counts 3–5 at both (p=45: 9/12/3 seeds with 3/4/5; p=47: 10/8/5).
- **Q5b holds (no enrichment).** At p=45, 41 of 90 key-frequency draws (45.6%) have gcd(k,45) > 1, against 45.4% expected
  from random sets of the same sizes (two-sided permutation p = 1.0). Power is limited: ~90 draws means a shift of
  roughly 10 points or more would be needed to see it.
- **Q5c placebo passes.** The same set of k at p=47 (where it means nothing): 45 of 87 (51.7%) vs 43.5% expected, p = 0.13.
  Not significant, so the test did not fire spuriously, but it is the biggest deviation from chance in the whole
  Q5 analysis and a reminder that p≈0.1 results turn up in null data.
- **Q5d holds.** Mean number of key frequencies: 3.75 (p=45) vs 3.78 (p=47), p = 1.0.
- Validity: restricted/full loss ratio is up to 4.7 (p=45) and 2.7 (p=47), under my 10× stop rule. These occur where
  the full loss is already tiny (seed 19 at p=45: 1.65e-6 vs 7.8e-6 restricted); the largest absolute restricted
  loss is 6e-3 (chance level is about 3.8), and excluded loss ≥ 4.7 on every seed.
- Both sweeps grok fast with 60% training data: the median seed has ≥50% test accuracy by the first logged step (250).
- **Honest assessment of the prediction record:** held: Q1, Q2, Q3, Q5a–d; failed: Q4. Most of my predictions were
  "nothing special happens" (no favoured frequencies, no composite effect), which are easy to satisfy and
  are not strong tests. The two predictions that could have been surprising were Q3 (held, clearly) and Q4 (failed).

## 2026-10-04 ~04:00 CDT — Why does the trig-identity fit fail? (Phase 1 model, pre-registered hypotheses)

Open question from Phase 1 claim 10: Σₖ [aₖ cos + bₖ sin](wₖ(a+b−c)) + class bias gives R² = 0.82 and 66.7% argmax accuracy.
Written **before** looking at the 3-D Fourier decomposition of the logits (in the orthonormal basis, projecting
onto a subset of components *is* the least-squares fit, so energy fractions are exact):
- **H1 (same frequency, other sign patterns).** The logits contain extra components at the *same* key frequency in all three
  axes (e.g. cos(w(a+b+c)), i.e. an "elliptical" rather than circular readout: A·cos(ws)cos(wc) + B·sin(ws)sin(wc) with
  A≠B contains both cos(w(s−c)) and cos(w(s+c))). Prediction: adding all 8 real (k,k,k) components per key frequency
  lifts argmax accuracy to ≳ 95%.
- **H2 (cross-frequency / harmonic terms).** The missing energy sits at components mixing different frequencies
  (e.g. (k₁,k₁,k₂), (k,k,2k)). Prediction: if H1 is false, most of the remaining energy is in cross-frequency terms.
- **H3 (amplitude depends on (a,b)).** Not directly testable in this decomposition; it would show up as many small
  components spread over (a,b) frequencies other than the key ones. Will be reported only if H1 and H2 together leave
  > 10% of class-varying energy unexplained.
My expectation: H1 is the main reason (~60% confidence), H2 second.
Script: `scripts/phase1_trigfit_gap.py` → `runs/p113_s0/trigfit_gap.json`.

### Result: the trig-fit gap (`runs/p113_s0/trigfit_gap.json`, `trigfit_gap_control.json`; scripts `phase1_trigfit_gap.py`)

3-D Fourier decomposition of the Phase 1 logits over (a, b, c), as fractions of the class-varying energy:
class bias (0,0,k) 23.0%; a, b and c all at the same key frequency (all 8 sign patterns) 67.8%; a, b at key k with c at a
*different key* frequency 0.8%; a, b at key k with c at a **non-key** frequency 6.9%; everything else 1.4%.
- **H1 refuted.** Bias + all 8 same-frequency sign patterns reconstruct only 61.7% of argmax decisions (predicted ≳ 95%).
- **H2 refuted as stated.** Key-to-key cross-frequency terms are 0.8% of the energy.
- **Energy is a poor guide to decisions here.** The logits are large (−195 to +64), so the top few terms hold most of the
  energy but the argmax depends on fine detail: ranked by energy, 16 components give 51% accuracy, 64 give 87%,
  256 give 95%, and 512 give 99.98% (out of 1.44 million).
- **The 6.9% block is mostly at harmonics and combinations of the key frequencies** (c-axis, with a and b at key k):
  2×24 (48), 24−46 (22), 24+46 (43), 2×46 (21), 3×24 (41), 24−28, 24+28, 4×24. This is the pattern a ReLU MLP
  would produce by mixing frequencies. (Seen first in the top-12 list, then tested; **exploratory, not pre-registered**.)
- **Test.** Letting c use the 4 key frequencies plus 20 harmonic/combination frequencies (m×k for m=2,3,4 and k₁±k₂, folded
  mod 113) gives 98.4% argmax accuracy (CE 0.035) with a, b restricted to the key frequencies. Control: 40 random draws
  of 20 extra non-key c-frequencies give mean 80.3%, min 64.6%, max 94.1%; key frequencies alone give 63.4%.
  So the structured set beats all 40 random sets (p ≲ 1/41), but **adding any extra components helps a lot**
  (random mean 80% vs 63%), so the structured effect is the ~18-point gap to random, not the whole improvement.
- **What this changes in the README.** Claim 10 is rewritten: the simple "Σ cos(w(a+b−c))" form is an energy-level
  description that misses what the argmax needs. I have not shown *how* the network produces these extra terms, or that
  they are a "sharpening" of the peak (a plausible but untested reading).
