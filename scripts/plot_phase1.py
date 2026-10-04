"""Figure for Phase 1: grokking curves and Fourier progress measures. Usage: python scripts/plot_phase1.py runs/p113_s0"""
import sys, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

run = sys.argv[1]
log, prog = json.load(open(f"{run}/log.json")), json.load(open(f"{run}/progress.json"))
BLUE, ORANGE, AQUA, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"
INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF})
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
s = [r["step"] for r in log]
a.plot(s, [r["train_loss"] for r in log], color=BLUE, lw=2, label="train")
a.plot(s, [r["test_loss"] for r in log], color=ORANGE, lw=2, label="test")
ps = [r["step"] for r in prog]
b.plot(ps, [r["test_full"] for r in prog], color=ORANGE, lw=2, label="test loss (full)")
b.plot(ps, [r["test_restricted"] for r in prog], color=AQUA, lw=2, ls="-.", label="restricted to key freqs")
b.plot(ps, [r["test_excluded"] for r in prog], color=MAGENTA, lw=2, ls="--", label="key freqs removed")
for ax, t in ((a, "Memorise, then generalise"), (b, "Fourier progress measures (test split)")):
    ax.set_yscale("log"); ax.set_xlabel("training step"); ax.set_ylabel("cross-entropy loss")
    ax.set_title(t, loc="left", color=INK, fontsize=11)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e5e4e0", lw=0.8); ax.legend(frameon=False, loc="upper right" if ax is a else "lower left")
fig.tight_layout(); fig.savefig("docs/phase1_progress.png", dpi=150)
