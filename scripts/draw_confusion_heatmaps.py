import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


CSI = np.array(
    [
        [0.654, 0.445, 0.412, 0.246, 0.128],
        [0.635, 0.425, 0.395, 0.230, 0.115],
        [0.550, 0.350, 0.310, 0.160, 0.075],
        [0.600, 0.400, 0.370, 0.210, 0.100],
        [0.590, 0.390, 0.355, 0.190, 0.090],
        [0.615, 0.410, 0.380, 0.220, 0.110],
    ]
)

POD = np.array(
    [
        [0.825, 0.685, 0.638, 0.534, 0.224],
        [0.805, 0.665, 0.615, 0.510, 0.210],
        [0.710, 0.550, 0.480, 0.380, 0.130],
        [0.770, 0.630, 0.575, 0.475, 0.180],
        [0.760, 0.620, 0.560, 0.460, 0.170],
        [0.785, 0.645, 0.590, 0.490, 0.195],
    ]
)

ROWS = ["Full", "w/o CR", "w/o PWV", "w/o Omni-Scan", "w/o Spectral Loss", "w/ Cross-Gate"]
COLS = ["drizzle\n(0,2.5]", "light\n(2.5,8.0]", "moderate\n(8.0,16.0]", "heavy\n(16.0,30.0]", "torrential\n(30.0,)"]


def _heatmap(ax, data, title):
    im = ax.imshow(data, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks(range(len(COLS)))
    ax.set_xticklabels(COLS)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels(ROWS)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_xlim(-0.5, data.shape[1] - 0.5)
    ax.set_ylim(data.shape[0] - 0.5, -0.5)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", color="black", fontsize=9)
    return im


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im1 = _heatmap(axes[0], CSI, "CSI Scores")
    im2 = _heatmap(axes[1], POD, "POD Scores")

    cbar = fig.colorbar(im2, ax=axes, fraction=0.035, pad=0.04)
    cbar.set_label("Score")

    plt.tight_layout()
    plt.savefig("csi_pod_heatmaps.png", dpi=300)


if __name__ == "__main__":
    main()
