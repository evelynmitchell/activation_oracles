import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
EXPERIMENTS_DIR = "experiments"
DATA_DIR = "classification_eval_Qwen3-8B_single_token"
INPUT_FOLDER = f"{EXPERIMENTS_DIR}/{DATA_DIR}/"

IMAGE_FOLDER = "images"
CLS_IMAGE_FOLDER = f"{IMAGE_FOLDER}/classification_eval"
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(CLS_IMAGE_FOLDER, exist_ok=True)

OUTPUT_PATH = f"{CLS_IMAGE_FOLDER}/classification_results_{DATA_DIR}.png"

# Dataset groupings
IID_DATASETS = [
    "geometry_of_truth",
    "relations",
    "sst2",
    "md_gender",
    "snli",
    "ner",
    "tense",
]

OOD_DATASETS = [
    "ag_news",
    "language_identification",
    "singular_plural",
]


def calculate_accuracy(records, dataset_ids):
    """Calculate accuracy for specified datasets."""
    correct = 0
    total = 0

    for record in records:
        if record["dataset_id"] in dataset_ids:
            total += 1
            if record["ground_truth"] == record["target"]:
                correct += 1

    if total == 0:
        return 0.0
    return correct / total


def load_results_from_folder(folder_path):
    """Load all JSON results from folder and calculate accuracies."""
    folder = Path(folder_path)
    results = {}

    for json_file in sorted(folder.glob("*.json")):
        with open(json_file, "r") as f:
            data = json.load(f)

        # Extract a clean name for the lora
        lora_name = json_file.stem.replace("classification_results_lora_", "")

        records = data.get("records", [])

        # Calculate accuracies
        iid_acc = calculate_accuracy(records, IID_DATASETS)
        ood_acc = calculate_accuracy(records, OOD_DATASETS)

        results[lora_name] = {
            "iid_accuracy": iid_acc,
            "ood_accuracy": ood_acc,
        }

        print(f"{lora_name}:")
        print(f"  IID Accuracy: {iid_acc:.2%}")
        print(f"  OOD Accuracy: {ood_acc:.2%}")

    return results


def plot_accuracies(results, output_path=None):
    """Create bar plots for IID and OOD accuracies."""
    lora_names = list(results.keys())
    iid_accs = [results[name]["iid_accuracy"] for name in lora_names]
    ood_accs = [results[name]["ood_accuracy"] for name in lora_names]

    # Create shortened labels for better readability
    short_labels = [
        name.replace("checkpoints_", "").replace("Qwen3-8B", "").replace("_", " ").strip() for name in lora_names
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # IID plot
    x_pos = np.arange(len(lora_names))
    bars1 = ax1.bar(x_pos, iid_accs, color="steelblue", alpha=0.8)
    ax1.set_xlabel("LoRA Adapter", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax1.set_title("IID Dataset Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(short_labels, rotation=45, ha="right")
    ax1.set_ylim([0, 1])
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.1%}", ha="center", va="bottom", fontsize=9)

    # OOD plot
    bars2 = ax2.bar(x_pos, ood_accs, color="coral", alpha=0.8)
    ax2.set_xlabel("LoRA Adapter", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax2.set_title("OOD Dataset Accuracy", fontsize=14, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(short_labels, rotation=45, ha="right")
    ax2.set_ylim([0, 1])
    ax2.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.1%}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\nPlot saved to: {output_path}")

    plt.show()


print(f"Loading results from: {INPUT_FOLDER}\n")
results = load_results_from_folder(INPUT_FOLDER)

if not results:
    print("No JSON files found in the specified folder!")

print(f"\nGenerating plots...")
plot_accuracies(results, OUTPUT_PATH)
