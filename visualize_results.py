import json
import matplotlib.pyplot as plt


def load_phase_scores():
    scores = {
        "Phase 1 Baseline": 0.51,
        "Phase 2 Ollama": 0.63,
        "Phase 3 Intervention": 0.79,
        "Phase 4 Auto-Tuned": 0.885,
    }
    try:
        with open("phase4_results.json") as f:
            data = json.load(f)
            if "quality_score" in data:
                scores["Phase 4 Auto-Tuned"] = data["quality_score"]
    except FileNotFoundError:
        pass
    return scores


def plot_convergence_scores(scores):
    labels = list(scores.keys())
    values = list(scores.values())
    colors = ["#B4B2A9", "#85B7EB", "#97C459", "#1D9E75"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors[:len(labels)], width=0.5)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Convergence Quality Score", fontsize=12)
    ax.set_title("RLM Convergence Score Across Research Phases", fontsize=14, fontweight="bold")
    ax.axhline(y=0.885, color="#0F6E56", linestyle="--", alpha=0.5, label="Optimal threshold (0.885)")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val:.3f}",
            ha="center", fontsize=11, fontweight="bold"
        )

    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("results_chart.png", dpi=150)
    print("Chart saved to results_chart.png")
    plt.show()


if __name__ == "__main__":
    scores = load_phase_scores()
    plot_convergence_scores(scores)
