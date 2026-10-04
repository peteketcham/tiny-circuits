# tiny-circuits

Training small transformers from scratch on algorithmic tasks and taking them apart, on CPU.
Every claim below links to the evidence for it; failures and dead ends are in
[`notebook/LAB_NOTEBOOK.md`](notebook/LAB_NOTEBOOK.md).

**Status: Phase 1 (reproduce known results), one seed.** Phase 2 (extensions) has not started.

![Phase 1 progress](docs/phase1_progress.png)

## Reproduce

```bash
pip install "jax[cpu]" numpy matplotlib optax
python -m tiny_circuits.train --p 113 --seed 0 --steps 20000 --ckpt_every 250 --out runs/p113_s0
#   (~40 min on 2 cores; add --max_seconds N and --resume to run in chunks. Resume is bit-exact.)
PYTHONPATH=. python scripts/phase1_analysis.py runs/p113_s0   # -> analysis.json, progress.json
PYTHONPATH=. python scripts/phase1_mechanism.py runs/p113_s0  # -> mechanism.json
PYTHONPATH=. python scripts/plot_phase1.py runs/p113_s0       # -> docs/phase1_progress.png
```

Model: 1 layer, 4 heads, d_model 128, d_mlp 512 (ReLU), no LayerNorm, learned positions, ~220k params.
Task: (a + b) mod 113, 30% of the 12,769 pairs for training, AdamW lr 1e-3, weight decay 1.0, full batch.

## Phase 1 claims (seed 0 only)

| # | Claim | Evidence | Confidence |
|---|-------|----------|------------|
| 1 | The model memorises, then groks. Train acc is 100% by step ~150; test acc is ~4% at step 4000, passes 50% at step 8600 and is 100% by step 10000. Weight norm falls from 58.8 (step 4000) to 37.2 (step 20000). | `runs/p113_s0/log.json` | High for this seed; not yet tested on others |
| 2 | The learned algorithm uses exactly four key frequencies, k ∈ {24, 28, 46, 56}. | Embedding Fourier norms for these are 7.7, 7.7, 6.8, 6.0; the next is 2.5 (k=1) then <0.7. 93% of embedding norm² is in these four. Logit energy that varies with the output class is concentrated on the (k,k) pairs of the same four (24: 39%, 46: 28%, 28: 5%, 56: 3.5%, plus a large class-dependent constant, 23%). `analysis.json` | High for this seed |
| 3 | These frequencies are *sufficient and necessary at the logit level*. Test loss with only the key components (plus the constant) is 1.6e-6, against 5.8e-6 for the full model. With them removed it is 26.5. Twenty random 4-frequency sets give restricted loss ≥ 25.8 and excluded loss ≤ 7.8e-6. | `analysis.json` (`loss_test`, `control_random_freqs_test`) | High that the logits are built from these components; this does not by itself show *how* the weights compute them |
| 4 | All four attention heads matter. Removing any one raises test loss from 6e-6 to between 2.9 and 4.1 (chance is 4.73). | `analysis.json` (`head_ablation_test_loss`); ablation zeroes the head's attention pattern, which is crude | Medium |
| 5 | MLP neurons fall into four groups by dominant frequency (133 / 102 / 143 / 134 neurons for k = 24 / 28 / 46 / 56). **Not supported:** that neurons are cleanly single-frequency. Median variance fraction explained by the dominant frequency is only 0.22 and no neuron exceeds 0.9, as expected for ReLU neurons with harmonics. | `analysis.json` (`neuron_*`) | Medium for the grouping; the "single-frequency" reading is not claimed |
| 6 | Training is bit-for-bit reproducible on one machine, including across the chunked resume: the interrupted run and the resumed run agree exactly on train and test loss at steps 2200 and 4000. | Lab notebook, 2026-10-03 | High on this machine; other hardware untested |
| 7 | The input and output sides use the same four frequencies. The top-4 Fourier frequencies of both `W_E` and `W_U` are {24, 28, 46, 56}. (80% of `W_U` norm² is in them, against 93% for `W_E`.) | `mechanism.json` | High for this seed |
| 8 | The heads specialise by frequency, in two near-duplicate pairs. Heads 0 and 2 write mostly k=46 (81% of key-frequency norm², with 13% on 28 and 6% on 56). Heads 1 and 3 write mostly k=24 (86%). This is what each head's value-output circuit *writes*, not proof of how the MLP uses it. | `mechanism.json` (`head_OV_frequency_content`) | Medium |
| 9 | Activation patching (3000 clean/corrupted pairs; metric = how often the *clean* answer comes back; chance 0.9%): no single head's clean output restores it (1–2%); all four heads restore 100%; patching all heads except one restores only ~32–34% whichever one is left out; patching all MLP activations restores 100%; patching only one frequency group of neurons restores 1.6–5.3%. So the answer is computed jointly from all four heads and all four neuron groups, not by any single sub-circuit. | `mechanism.json` (`activation_patching_clean_answer_accuracy`) | Medium |
| 10 | **Partly refuted:** the textbook form logit(c) ≈ Σₖ aₖcos(wₖ(a+b−c)) + bₖsin(wₖ(a+b−c)) + class bias does *not* describe this model well enough. It explains R² = 0.82 of logit variance (0.59 without the class bias), and the fitted logits pick the right answer for only 66.7% of pairs. The model's real logits have additional structure that I have not characterised. | `mechanism.json` (`trig_fit_*`) | Medium-high that the simple form is incomplete; cause unknown |

## Not done yet (in the Phase 1 plan)

- **Why claim 10 fails.** The fit is on raw logits, which are dominated by large values; the gap may come from amplitude varying with (a, b), cross-frequency terms, or ReLU harmonics. Not yet tested.
- **Weight-level circuit.** Claim 3 is about the logits. I have not shown, from the weights alone, how the MLP produces the key-frequency terms.
- **Seeds.** Everything is one seed. Seed universality is the first Phase 2 experiment.
- **Observed, not investigated:** training loss shows regular sharp spikes every ~1.4k steps (visible in the figure, left panel).

## Layout

```
tiny_circuits/   model.py (JAX transformer), data.py, train.py, analysis.py (Fourier toolkit)
scripts/         phase1_analysis.py, phase1_mechanism.py, plot_phase1.py
runs/p113_s0/    config, training log, final weights, analysis + progress-measure JSON
notebook/        LAB_NOTEBOOK.md
```

Per-checkpoint weights (`ckpt_*.npz`, `state.pkl`) are not committed; rerun training to regenerate them.
