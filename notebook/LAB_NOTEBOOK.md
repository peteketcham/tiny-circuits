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
- p=53, train_frac 0.5: started, then interrupted by the user at about step 2000; no conclusion.
  The partial log is in `runs/sweep_p53/s0/` and should not be read as a result.
