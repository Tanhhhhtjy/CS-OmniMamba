import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.use("Agg")


ROWS = [
    "Truth",
    "Full",
    "w/o CR",
    "w/o PWV",
    "w/o Omni-Scan",
    "w/o Spectral-Loss",
    "w/o Cross-Gate",
    "ConvLSTM",
    "TrajGRU",
    "itransformer",
]
COLS = ["T+1h", "T+2h", "T+3h"]


def load_image(path):
    img = Image.open(path).convert("RGBA")
    return np.array(img)


def main():
    fig, axes = plt.subplots(len(ROWS), len(COLS), figsize=(4.2, 9.0))
    plt.subplots_adjust(left=0.18, right=0.88, top=0.94, bottom=0.06, wspace=0.05, hspace=0.15)

    img_idx = 1
    for i, row_name in enumerate(ROWS):
        for j, col_name in enumerate(COLS):
            ax = axes[i, j]
            path = f"img_{img_idx:02d}.png"
            img = load_image(path)
            ax.imshow(img)
            ax.axis("off")
            if i == 0:
                ax.set_title(col_name, fontsize=10)
            if j == 0:
                ax.text(
                    -0.28,
                    0.5,
                    row_name,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=9,
                )
            img_idx += 1

    cbar_ax = fig.add_axes([0.9, 0.08, 0.03, 0.84])
    gradient = np.linspace(0, 30, 256).reshape(-1, 1)
    cbar_ax.imshow(gradient, aspect="auto", cmap="jet", origin="lower")
    cbar_ax.set_yticks([0, 64, 128, 192, 255])
    cbar_ax.set_yticklabels(["0", "5", "10", "20", "30"])
    cbar_ax.set_xticks([])
    cbar_ax.set_title("mm/h", fontsize=9)

    plt.savefig("panel_grid.png", dpi=300)


if __name__ == "__main__":
    main()
