import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    "Angry":        [[ 0.40,-0.11,-0.21, 0.00],[-0.12, 0.21, 0.01, 0.16],[-0.06, 0.08, 0.12, 0.24],[-0.07, 0.06, 0.08, 0.26]],
    "Anticipation": [[ 0.48, 0.07, 0.02, 0.01],[-0.05, 0.23, 0.09, 0.14],[-0.04, 0.05, 0.14, 0.11],[-0.06, 0.15, 0.16, 0.19]],
    "Disgust":      [[ 0.34,-0.09,-0.23,-0.43],[-0.19, 0.15, 0.05, 0.09],[-0.12, 0.05, 0.14, 0.06],[-0.11, 0.06, 0.05, 0.08]],
    "Fear":         [[ 0.18, 0.00,-0.10,-0.29],[-0.03, 0.06, 0.06, 0.03],[-0.02, 0.16, 0.17, 0.08],[-0.16,-0.01,-0.04, 0.04]],
    "Happy":        [[ 0.45,-0.10,-0.32,-0.08],[-0.11, 0.16, 0.06, 0.18],[-0.04, 0.02, 0.09, 0.19],[-0.13, 0.08, 0.08, 0.22]],
    "Neutral":      [[ 0.46,-0.14,-0.25,-0.11],[-0.04, 0.26, 0.09, 0.22],[-0.05, 0.11, 0.15, 0.24],[-0.18, 0.08, 0.08, 0.28]],
    "Sad":          [[ 0.45,-0.05,-0.25, 0.23],[-0.07, 0.23, 0.07, 0.22],[ 0.06, 0.10, 0.12, 0.29],[-0.06, 0.08, 0.09, 0.27]],
    "Surprise":     [[ 0.48,-0.07,-0.40,-0.01],[-0.09, 0.17,-0.00, 0.16],[-0.09, 0.00, 0.05, 0.18],[-0.18, 0.03,-0.00, 0.23]],
    "Trust":        [[ 0.43,-0.04,-0.20,-0.26],[-0.17, 0.26, 0.05, 0.18],[-0.08, 0.13, 0.14, 0.18],[-0.13, 0.11, 0.13, 0.21]],
}

datasets = ["EmoLit", "Emotion20", "GoEmotions", "XED_SMED"]

matrices = np.array([data[cls] for cls in data])  # shape: (9, 4, 4)
avg_matrix = matrices.mean(axis=0)                 # shape: (4, 4)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    avg_matrix,
    ax=ax,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    vmin=-1, vmax=1,
    xticklabels=datasets,
    yticklabels=datasets,
    linewidths=0.5,
    linecolor="white",
)
ax.set_title("Average pearson(a_1, a_n\\1) across all emotions", fontsize=12)
ax.set_xlabel("a_n\\1 (leave-one-out)")
ax.set_ylabel("a_1 (individual)")
plt.tight_layout()
plt.savefig("avg_pearson.png", dpi=150)
plt.show()