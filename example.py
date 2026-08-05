"""Demonstrate how softmax weights produce a weighted average."""

from pathlib import Path

import matplotlib
import numpy as np

from softmax import softmax

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


VALUES = np.array([10.0, 20.0, 30.0])
LOGITS = np.array([1.0, 2.0, 3.0])
PLOT_PATH = Path(__file__).parent / "assets" / "softmax_weighted_average.png"


def save_plot(
    values: np.ndarray,
    logits: np.ndarray,
    weights: np.ndarray,
    output_path: Path,
) -> None:
    """Save a chart of softmax weights and weighted contributions."""
    item_labels = [f"Item {index}" for index in range(1, len(values) + 1)]
    score_labels = [
        f"{item}\nlogit = {logit:g}" for item, logit in zip(item_labels, logits)
    ]
    contributions = weights * values

    figure, (weight_axis, contribution_axis) = plt.subplots(
        1, 2, figsize=(10, 4.5), constrained_layout=True
    )
    colors = ["#8ecae6", "#219ebc", "#023047"]

    weight_bars = weight_axis.bar(score_labels, weights, color=colors)
    weight_axis.set_title("Logits become softmax weights")
    weight_axis.set_ylabel("Weight")
    weight_axis.set_ylim(0, 0.75)
    weight_axis.bar_label(weight_bars, labels=[f"{weight:.1%}" for weight in weights])
    contribution_bars = contribution_axis.bar(
        item_labels, contributions, color=colors
    )
    contribution_axis.set_title("Each weight scales its value")
    contribution_axis.set_ylabel("Weighted contribution")
    contribution_axis.set_ylim(0, 22)
    contribution_axis.bar_label(
        contribution_bars,
        labels=[
            f"{weight:.3f} × {value:g}\n= {result:.2f}"
            for weight, value, result in zip(weights, values, contributions)
        ],
    )

    figure.suptitle(
        f"The contributions sum to a weighted average of {contributions.sum():.2f}",
        fontsize=13,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    """Run the example and generate its visualization."""
    weights = softmax(LOGITS)
    weighted_average = np.sum(weights * VALUES)

    save_plot(VALUES, LOGITS, weights, PLOT_PATH)

    print(f"Values:                {VALUES}")
    print(f"Logits:                {LOGITS}")
    print(f"Weights:               {np.round(weights, 4)}")
    print(f"Sum of weights:        {weights.sum():.4f}")
    print(f"Weighted average:      {weighted_average:.2f}")
    relative_plot_path = PLOT_PATH.relative_to(Path(__file__).parent).as_posix()
    print(f"Plot saved to:         {relative_plot_path}")


if __name__ == "__main__":
    main()
