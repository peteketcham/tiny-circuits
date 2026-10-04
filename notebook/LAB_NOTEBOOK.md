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
  on every seed (restricted min 7–21 vs ≲2e-5 for the key set). Key-frequency counts: 3–5 (four seeds with 4, two with 3, one with 5... see `summary_c1.json`).
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
