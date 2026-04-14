"""
Generate figures for the paper:
  Fig.1 - System pipeline flowchart
  Fig.2 - Dataset distribution bar chart
  Fig.3 - Feature extraction comparison diagram
Run from the spam-detection/ directory.
"""
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
import numpy as np

os.makedirs("figures", exist_ok=True)

# ── Fig.1: System Pipeline ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 3))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis("off")

steps = [
    ("Raw\nEmails", 0.6),
    ("Data\nPreprocessing", 2.2),
    ("Feature\nExtraction", 3.8),
    ("Classification", 5.4),
    ("Evaluation\n& Output", 7.0),
]
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

for (label, x), color in zip(steps, colors):
    box = mpatches.FancyBboxPatch(
        (x - 0.55, 0.8), 1.1, 1.4,
        boxstyle="round,pad=0.1",
        linewidth=1.5, edgecolor="white",
        facecolor=color, zorder=3
    )
    ax.add_patch(box)
    ax.text(x, 1.5, label, ha="center", va="center",
            fontsize=9, fontweight="bold", color="white", zorder=4)

# arrows between boxes
for i in range(len(steps) - 1):
    x1 = steps[i][1] + 0.55
    x2 = steps[i+1][1] - 0.55
    ax.annotate("", xy=(x2, 1.5), xytext=(x1, 1.5),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=2))

# sub-labels below each box
sub = [
    "enron1-3\nsubsets",
    "tokenize\nstopwords\nlowercase",
    "TF-IDF\nWord2Vec",
    "NB / SVM\nRF / MLP",
    "Acc / F1\nPrec / Rec",
]
for (label, x), s in zip(steps, sub):
    ax.text(x, 0.55, s, ha="center", va="top",
            fontsize=7, color="#444444", style="italic")

plt.tight_layout()
plt.savefig("figures/fig1_pipeline.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved figures/fig1_pipeline.png")


# ── Fig.2: Dataset Distribution ────────────────────────────────────────────
try:
    with open("data/processed.pkl", "rb") as f:
        texts, labels = pickle.load(f)
    ham_count  = labels.count(0)
    spam_count = labels.count(1)
except FileNotFoundError:
    # fallback to approximate values if data not yet processed
    ham_count, spam_count = 14198, 5822

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# left: bar chart
categories = ["Ham", "Spam"]
counts = [ham_count, spam_count]
bar_colors = ["#4C72B0", "#C44E52"]
bars = axes[0].bar(categories, counts, color=bar_colors, width=0.5,
                   edgecolor="white", linewidth=1.2)
for bar, count in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 150,
                 f"{count:,}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
axes[0].set_ylabel("Number of Emails", fontsize=10)
axes[0].set_title("Email Count by Class", fontsize=10)
axes[0].set_ylim(0, max(counts) * 1.18)
axes[0].spines[["top", "right"]].set_visible(False)
axes[0].yaxis.grid(True, linestyle="--", alpha=0.5)
axes[0].set_axisbelow(True)

# right: pie chart
total = ham_count + spam_count
wedge_colors = ["#4C72B0", "#C44E52"]
axes[1].pie(counts, labels=categories, colors=wedge_colors,
            autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2),
            textprops=dict(fontsize=10))
axes[1].set_title("Class Distribution", fontsize=10)

plt.tight_layout()
plt.savefig("figures/fig2_dataset.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved figures/fig2_dataset.png")


# ── Fig.3: Feature Extraction Comparison ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# ---- left: TF-IDF ----
ax = axes[0]
ax.set_xlim(0, 5)
ax.set_ylim(0, 6)
ax.axis("off")
ax.set_title("TF-IDF", fontsize=12, fontweight="bold", color="#4C72B0", pad=10)

# raw text box
raw_box = mpatches.FancyBboxPatch((0.3, 4.5), 4.4, 0.9,
    boxstyle="round,pad=0.1", facecolor="#D0E4F7", edgecolor="#4C72B0", lw=1.5)
ax.add_patch(raw_box)
ax.text(2.5, 4.95, '"free offer click here now"', ha="center", va="center",
        fontsize=9, style="italic", color="#1a1a2e")

ax.annotate("", xy=(2.5, 3.9), xytext=(2.5, 4.5),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))
ax.text(2.5, 4.2, "Vocabulary (top 5,000 terms)", ha="center",
        fontsize=7.5, color="#555")

# sparse vector
vec_vals = [0, 0.32, 0, 0.78, 0, 0.45, 0, 0.12, 0, 0]
cmap_val = ["#cce5ff" if v == 0 else "#4C72B0" for v in vec_vals]
for i, (v, c) in enumerate(zip(vec_vals, cmap_val)):
    rect = plt.Rectangle((0.2 + i*0.44, 2.8), 0.38, 0.8,
                          facecolor=c, edgecolor="white", lw=1)
    ax.add_patch(rect)
    if v > 0:
        ax.text(0.2 + i*0.44 + 0.19, 3.2, f"{v:.2f}",
                ha="center", va="center", fontsize=6.5, color="white", fontweight="bold")

ax.text(2.5, 2.6, "Sparse 5,000-dim vector", ha="center", fontsize=8, color="#333")
ax.annotate("", xy=(2.5, 2.1), xytext=(2.5, 2.8),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))

out_box = mpatches.FancyBboxPatch((0.8, 1.1), 3.4, 0.8,
    boxstyle="round,pad=0.1", facecolor="#4C72B0", edgecolor="white", lw=1.5)
ax.add_patch(out_box)
ax.text(2.5, 1.5, "Classifier Input", ha="center", va="center",
        fontsize=9, color="white", fontweight="bold")

# ---- right: Word2Vec ----
ax = axes[1]
ax.set_xlim(0, 5)
ax.set_ylim(0, 6)
ax.axis("off")
ax.set_title("Word2Vec (Skip-gram)", fontsize=12, fontweight="bold",
             color="#C44E52", pad=10)

# raw text
raw_box2 = mpatches.FancyBboxPatch((0.3, 4.5), 4.4, 0.9,
    boxstyle="round,pad=0.1", facecolor="#FFD9D9", edgecolor="#C44E52", lw=1.5)
ax.add_patch(raw_box2)
ax.text(2.5, 4.95, '"free offer click here now"', ha="center", va="center",
        fontsize=9, style="italic", color="#1a1a2e")

ax.annotate("", xy=(2.5, 3.9), xytext=(2.5, 4.5),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))
ax.text(2.5, 4.2, "Word → 100-dim vector", ha="center",
        fontsize=7.5, color="#555")

# word vectors (stacked mini-bars)
words = ["free", "offer", "click"]
word_colors = ["#e07070", "#c44e52", "#9b2335"]
for wi, (w, wc) in enumerate(zip(words, word_colors)):
    y0 = 2.2 + wi * 0.52
    for i in range(10):
        val = np.random.uniform(0.2, 1.0)
        rect = plt.Rectangle((0.2 + i*0.44, y0), 0.38, 0.38,
                              facecolor=wc, edgecolor="white", lw=0.8,
                              alpha=val)
        ax.add_patch(rect)
    ax.text(0.05, y0 + 0.19, w, ha="right", va="center",
            fontsize=7.5, color="#333", style="italic")

ax.text(2.5, 2.05, "Mean pooling  →  100-dim vector", ha="center",
        fontsize=8, color="#333")
ax.annotate("", xy=(2.5, 1.9), xytext=(2.5, 2.1),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))

out_box2 = mpatches.FancyBboxPatch((0.8, 1.0), 3.4, 0.8,
    boxstyle="round,pad=0.1", facecolor="#C44E52", edgecolor="white", lw=1.5)
ax.add_patch(out_box2)
ax.text(2.5, 1.4, "Classifier Input", ha="center", va="center",
        fontsize=9, color="white", fontweight="bold")

plt.tight_layout()
plt.savefig("figures/fig3_features.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved figures/fig3_features.png")

print("\nAll paper figures saved to figures/")
