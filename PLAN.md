# Phase 2 plan (written 2026-10-03 22:25 CDT, before any Phase 2 results)

Work window ends **08:00 CDT 2026-10-04**; final push by 07:45. Predictions below are fixed now and will not be
edited. If results disagree, the README reports the disagreement.

## Setup

p = 53, train_frac 0.5, 1 layer / 4 heads / d_model 128 / d_mlp 512, AdamW lr 1e-3 wd 1.0, full batch,
6000 steps, seeds 0..N-1 (seed controls both init and the train/test split). Same code as Phase 1.
The setting differs from Phase 1 (p=113, 30%), so Phase 2 results are not directly comparable to it.

## Questions and pre-registered predictions

**Q1. Does every seed find a Fourier circuit, and how many frequencies does it use?**
Prediction: ≥90% of seeds reach ≥99% test accuracy; each such seed has 2–7 key frequencies.
(Key frequency = (k,k) pair holding ≥1% of the class-varying logit energy; same definition as Phase 1.)

**Q2. Do seeds pick the same frequencies?**
Prediction: no. Chosen sets look like random draws from the 26 available frequencies: no frequency appears in
more than ~50% of seeds, and the most common pair of identical sets is rare. (Null model: uniform random sets
of the observed sizes. I will compare to it.)

**Q3. Can the final frequencies be predicted before the model generalises?**
Prediction: partly. Taking the top-n embedding frequencies at the last checkpoint where test accuracy is still
<20% (n = final number of key frequencies), mean overlap (Jaccard) with the final set > 0.5, versus a chance
level of roughly 0.1.

**Q4. Are heads duplicated as in the Phase 1 model?**
Prediction: in most seeds at least two heads write the same dominant frequency (share ≥50% of their key-frequency
norm² on one frequency, same frequency).

**Q5 (stretch, only if checkpoint C2 is clean). Composite modulus.** p = 105 = 3·5·7 vs a prime of similar size.
Prediction: composite does not change the Fourier story; key frequencies are not enriched for divisors of 105
(frequencies k with gcd(k,105) > 1). I have low confidence in this one.

## Checkpoints (each ends with a note in the lab notebook and a push)

- **C1 — after 8 seeds (~23:30 CDT).** Check: grok rate; hand-verify the analysis on two seeds by reading their
  curves and key-frequency tables; confirm the random-frequency control still separates (restricted ≈ full,
  excluded ≫ full, random sets fail) on every grokked seed. **Stop and fix** if any seed has restricted loss more than
  10× its full loss — that would mean the analysis, not the model, is wrong.
- **C2 — after all seeds (~01:30 CDT).** Tabulate against Q1–Q4 *as written above*. Re-train 3 seeds from scratch and
  require bit-identical final loss. Write down which predictions failed. Check that conclusions don't hinge on the
  1% key-frequency threshold (rerun with 0.5% and 2%).
- **C3 — ~02:00 CDT.** Decide whether to run Q5 based on C2. If C2 found problems, spend the time on them instead.
- **C4 — 07:00 CDT.** Freeze experiments. Re-run every analysis script from the committed files, diff against the
  committed JSON, update README claims and confidence levels, push. Stop by 07:45.

## Rules for myself

- Anything I change in the analysis after seeing a result goes in the notebook, with the reason (as in Phase 1).
- No claim in the README without a pointer to a JSON file in the repo.
- Failed or odd seeds stay in the tables; no cherry-picking.
- `main` is untouched. Work is on branch `phase2-seeds`; merging is the user's call.

---
## Addendum A (written 2026-10-04 01:35 CDT, after C2, before any Q5 run) — pre-registration of Q5

**Deviation from the plan above.** Q5 said p = 105 vs a prime near it. At ~0.2 s/step that allows only ~8 seeds in the
time left, too few for any statistical statement. Replaced with **p = 45 (= 3²·5, composite) vs p = 47 (prime)**,
train_frac 0.6, 4000 steps, seeds 0–23 for each. (Pilot, seed 0 only: both reach 100% test accuracy by step 1000.)

**Predictions, fixed now.**
- Q5a: ≥ 90% of seeds reach ≥ 99% test accuracy at both moduli, with 2–7 key frequencies each.
- Q5b (main): key frequencies of p = 45 models are **not** enriched or depleted for frequencies k with gcd(k,45) > 1
  (10 of the 22 frequencies, 45%). Two-sided permutation test on the pooled key-frequency count, permuting within
  seeds (random sets of the observed sizes), 5000 draws; "holds" means p > 0.05.
  My confidence is low (~50/50): there is a plausible mechanism for enrichment (short-period frequencies correspond
  to the subgroup structure) or depletion.
- Q5c: placebo. Same statistic for the p = 47 models using the *same set of k* ({3,5,6,9,10,12,15,18,20,21}), which has no
  meaning mod 47. Prediction: not enriched (p > 0.05). If this placebo fires, the test is unreliable and Q5b is void.
- Q5d: mean number of key frequencies differs between p = 45 and p = 47 by less than 1 (two-sided permutation test, p > 0.05).

---
## Addendum B (written 2026-10-04 04:05 CDT, before any p=113 sweep run) — does it transfer to the Phase 1 setting?

Phase 2 used small moduli and 50–60% training data. Check the headline findings at **p = 113, train_frac 0.3** (the Phase 1 setting),
seeds 1–6 (seed 0 is the Phase 1 run), 14000 steps, same code. Only ~6 seeds fit (≈30 min each), so this is a
robustness check, not a statistical test. Predictions, fixed now:
- T1: ≥ 5 of 6 seeds reach ≥ 99% test accuracy within 14000 steps. (Phase 1 seed 0 needed ~10000; some seeds may be slower.)
- T2: every grokked seed has 3–5 key frequencies (same ≥1% definition).
- T3: no two seeds (counting seed 0 = {24, 28, 46, 56}) have the same key-frequency set.
- T4: circuit validity holds as before: restricted loss < 10× full loss (or < 1e-3 absolute) and excluded loss > 10 on every grokked seed;
  random-frequency controls fail.
If fewer than 6 seeds finish before 07:15 CDT, I analyse the ones that did and report n.

---
## Addendum C (written 2026-10-04 08:40 CDT, before any multitask code or run) — one network, two tasks

Work window ends **17:30 CDT** (user allowed until 18:00); final push by 17:30.

**Question.** Train one network on both (a+b) mod p and (a·b) mod p, with the task given by the final token. Does it build separate circuits
or share them? Multiplication mod a prime p is addition in disguise: with a primitive root g, a = g^i for a ≠ 0, so a·b = g^(i+j). I expect the
network to learn **Fourier features over the discrete log** (period p−1) for multiplication, alongside ordinary Fourier features (period p) for addition.

**Setup.** p = 53 (p−1 = 52), 1 layer / 4 heads / d_model 128 / d_mlp 512, AdamW lr 1e-3, wd 1.0, full batch, train_frac 0.5 *per task*, seeds 0–11.
Three conditions, same code path, same hyperparameters: `add` (addition only), `mul` (multiplication only), `addmul` (both).
Step budget is set from a pilot (seed 0 only) and may not exceed 10000. Pairs involving 0 are included in training; the key-frequency
analysis for multiplication covers nonzero a, b, c only, and CE for it uses a softmax over the 52 nonzero classes (same definition for full and restricted).
Key frequencies use the same rule as before: (k,k) pair holding ≥ 1% of the class-varying logit energy, in the basis of the relevant cyclic group.

**Predictions, fixed now.**
- **M1.** `addmul` reaches ≥ 99% test accuracy on *both* tasks within the budget in ≥ 80% of seeds.
- **M2.** In grokked `addmul` seeds, addition uses 3–5 additive key frequencies and multiplication uses 3–5 key *log*-frequencies, and both pass the
  validity checks used so far (restricted loss < 10× full or < 1e-3 absolute; excluded loss > 5; random same-size frequency sets fail).
  Same for single-task `mul`. (Confidence: high that mul is Fourier in log-space; moderate on the exact counts.)
- **M3.** The two tasks' key features are not specially aligned in embedding space. For each additive key feature f_k (cos/sin of 2πka/p) and each
  multiplicative key feature h_j (cos/sin of 2πj·log a/(p−1)), take the embedding-space directions u = Eᵀf, v = Eᵀh (E = rows a = 1..p−1 of W_E).
  Let O_real = mean cos² between additive-key and multiplicative-key directions, and O_null = the same with the multiplicative features replaced by
  random *non-key* log-frequencies (same count, 200 draws). Prediction: O_real / O_null between 0.5 and 2.
- **M4.** Interference slows learning: median step at which test accuracy first exceeds 50% (log resolution 250 steps) is ≥ 25% larger for the addition task
  in `addmul` than in `add`, and likewise for multiplication in `addmul` vs `mul`. (Confidence ~60%; either direction would be informative.)
- **M5.** MLP neurons specialise by task. For each neuron, compare mean-squared post-ReLU activation (final position) over all addition inputs vs
  all multiplication inputs. In grokked `addmul` seeds, more than 50% of neurons have a ratio ≥ 5 in one direction.

**Rules.** As before: deviations and post hoc analyses are labelled in the notebook; failed seeds stay in the tables; `main` is untouched.

---
## Addendum D (written 2026-10-04 09:58 CDT, after M1 failed 0/4 at the registered setting) — post hoc deviation regime
M1 failed (see notebook). M2–M5 are only meaningful where `addmul` actually groks, so I add a **deviation regime, labelled post hoc**:
train_frac 0.7 per task (p=53, everything else unchanged), up to 8000 steps. Pilot: seed 0 only. If it reaches ≥99% on both tasks, run seeds 0–5 for
`addmul` at 0.7 plus `add` and `mul` single-task baselines at 0.7 (same seeds, 3000–4000 steps). If the pilot fails, try 0.85 once, then stop and report that
the two-task network did not grok in my budget. M1–M5 are then scored *in this regime* and reported as "deviation regime", never merged with the registered result.
Predictions M2–M5 stand as written in Addendum C (M4 now compares against baselines at 0.7). Seed count is smaller than registered (6 not 12) because of time.

**Addendum D result / D2 (written 10:46 CDT).** Pilot seed 0, 8000 steps: frac 0.7 → test acc 0.708/0.711 (add/mul), no grok. frac 0.85 → 0.893/0.867 at 8000 and still rising slowly.
D2 (post hoc): extend the 0.85 pilot to 20000 steps (same seed, bit-exact resume). If both tasks reach ≥99%, run seeds 1–3 at 0.85 for 20000 steps (≈35 min each) plus single-task baselines; otherwise stop the multitask line and report.

---
**Multitask scorecard (written ~12:15 CDT; see README Phase 3).** M1 **failed** (registered setting 0/4; pilots at 0.7 and 0.85 per-task data did not reach 99% on both).
M2: held for single-task mul (6/6, 3–5 log-frequencies, valid) and single-task add (6/6 valid; counts 4,6,4,4,5,4 so 5/6 inside 3–5). M3/M5: n=1, consistent with the predictions but not formally scored.
M4: not testable as registered (the 50% metric is degenerate); with a post hoc 90% threshold, addmul is ≥19× slower than single-task (n=1).

---
## Addendum E (written 2026-10-04 12:25 CDT, post hoc extension; D2's "otherwise stop" rule is overridden because time remains)
Run `addmul`, train_frac 0.85, 20000 steps, seeds 1 and 2 (seed 0 already done), to turn the n=1 descriptive results (P10, P13) into n=3.
Predictions fixed now: (a) each seed ends with ≥ 90% on both tasks (seed 0: 96%/99%); (b) neither reaches ≥ 99% on *both* tasks (seed 0 did not); (c) each task has 3–7 key frequencies and passes validity;
(d) M3 ratio between 0.5 and 2; (e) M5: >50% of neurons selective. Still exploratory (the regime was chosen after M1 failed).

**Addendum E result (written ~14:35 CDT).** (a) holds 3/3 (seed 0, 1, 2 all ≥ 96%/98.6% on both). (b) **fails**: seeds 1 and 2 reach ≥ 99% on both tasks (seed 2: 1.000/0.993). (c) holds: 4–6 key frequencies per task, all valid (the 3–7 range). (d) 2/3: ratios 0.80, 0.45, 0.57, seed 1 just under 0.5. (e) holds 3/3 (69–71% of neurons).
