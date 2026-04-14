"""
Generate confusion matrix plots from results.
Requires: data/results.json
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt

with open("data/results.json") as f:
    results = json.load(f)

os.makedirs("figures", exist_ok=True)

for r in results:
    cm = np.array(r["Confusion"])
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Ham","Spam"]); ax.set_yticklabels(["Ham","Spam"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{r['Classifier']} + {r['Feature']}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i,j], ha="center", va="center")
    plt.tight_layout()
    fname = f"figures/cm_{r['Classifier'].replace(' ','_')}_{r['Feature']}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")
