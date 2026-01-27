from typing import Iterable, Optional

import matplotlib.pyplot as plt


def plot_losses(train: Iterable[float], val: Iterable[float], save_dir: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(list(train), label="Train Loss")
    plt.plot(list(val), label="Val Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.savefig(f"{save_dir}/loss_curve.png")
    plt.close()


def plot_gate_history(history: Iterable[Optional[float]], save_dir: str) -> None:
    processed = []
    last = 0.5
    for h in history:
        if h is not None:
            last = h
        processed.append(last)

    plt.figure(figsize=(8, 4))
    plt.plot(processed)
    plt.title("Gate Value Evolution")
    plt.savefig(f"{save_dir}/gate_curve.png")
    plt.close()
