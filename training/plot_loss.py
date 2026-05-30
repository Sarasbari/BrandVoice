import os
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Resolve repository root
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)

def main():
    loss_curve_json = os.path.join(repo_root, "outputs", "loss_curve.json")
    output_png = os.path.join(repo_root, "outputs", "loss_curve.png")

    if not os.path.exists(loss_curve_json):
        print(f"Error: {loss_curve_json} not found. Please run the training script first.")
        return

    # 1. Read outputs/loss_curve.json
    with open(loss_curve_json, "r", encoding="utf-8") as f:
        loss_history = json.load(f)

    if not loss_history:
        print("Error: Loss history is empty.")
        return

    steps = [log["step"] for log in loss_history]
    losses = [log["loss"] for log in loss_history]
    final_loss = losses[-1]

    # Set dark background style
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Plot orange line
    ax.plot(steps, losses, color="#FF5A00", linewidth=2, label="Training Loss")

    # Add a horizontal dashed line at the final loss value
    ax.axhline(
        y=final_loss,
        color="#00FFCC",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        label=f"Final: {final_loss:.2f}"
    )

    # Style details
    ax.set_title("BrandVoice — QLoRA Training Loss (Notion Brand)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Training Step", fontsize=11, labelpad=10)
    ax.set_ylabel("Loss", fontsize=11, labelpad=10)

    # Grid lines at 0.5 intervals for Y-axis
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.grid(True, which="both", linestyle=":", color="#444444", alpha=0.7)

    # Legend
    ax.legend(loc="upper right", framealpha=0.8)

    # Tight layout to avoid cutting off labels
    plt.tight_layout()

    # Save to outputs/loss_curve.png at 150 DPI
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()

    print("Loss curve saved to outputs/loss_curve.png")

if __name__ == "__main__":
    main()
