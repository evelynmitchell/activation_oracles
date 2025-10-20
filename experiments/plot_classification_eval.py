import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
# Configuration
EXPERIMENTS_DIR = "experiments"
DATA_DIR = "classification_eval_Qwen3-32B_single_token"
INPUT_FOLDER = f"{EXPERIMENTS_DIR}/{DATA_DIR}/"

IMAGE_FOLDER = "images"
CLS_IMAGE_FOLDER = f"{IMAGE_FOLDER}/classification_eval"
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(CLS_IMAGE_FOLDER, exist_ok=True)

OUTPUT_PATH = f"{CLS_IMAGE_FOLDER}/classification_results_{DATA_DIR}.png"

# Filter out files containing any of these strings
FILTERED_FILENAMES = [
    "latentqa_only",
]

# Mapping from JSON filename to bar chart label
# Run script once to get the list of files, then fill in the labels
JSON_TO_LABEL = {
    "classification_results_lora_checkpoints_act_pretrain_cls_latentqa_mix_posttrain_Qwen3-32B.json": "Past / Future Lens -> Classification + LatentQA Posttrain",
    "classification_results_lora_checkpoints_act_pretrain_cls_only_posttrain_Qwen3-32B.json": "Past / Future Lens -> Classification Only Posttrain",
    "classification_results_lora_checkpoints_classification_only_Qwen3-32B.json": "Past / Future Lens -> Classification Only",
    "classification_results_lora_checkpoints_latentqa_only_Qwen3-32B.json": "Past / Future Lens -> LatentQA Only",
}

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
        return 0.0, 0
    return correct / total, total


def calculate_confidence_interval(accuracy, n, confidence=0.95):
    """Calculate binomial confidence interval for accuracy."""
    if n == 0:
        return 0.0

    # Use normal approximation for binomial confidence interval
    z_score = 1.96  # 95% confidence
    se = np.sqrt(accuracy * (1 - accuracy) / n)
    margin = z_score * se

    return margin


def load_results_from_folder(folder_path):
    """Load all JSON results from folder and calculate accuracies."""
    folder = Path(folder_path)
    results = {}

    json_files = sorted(folder.glob("*.json"))

    # Filter out files based on FILTERED_FILENAMES
    if FILTERED_FILENAMES:
        json_files = [f for f in json_files if not any(filter_str in f.name for filter_str in FILTERED_FILENAMES)]

    # Print dictionary template for easy copy-paste
    print("Found JSON files:")
    file_dict = {f.name: "" for f in json_files}
    print(file_dict)
    print()

    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        # Use label from mapping, or filename if not mapped
        label = JSON_TO_LABEL.get(json_file.name, json_file.name)

        records = data.get("records", [])

        # Calculate accuracies and counts
        iid_acc, iid_count = calculate_accuracy(records, IID_DATASETS)
        ood_acc, ood_count = calculate_accuracy(records, OOD_DATASETS)

        # Calculate confidence intervals
        iid_ci = calculate_confidence_interval(iid_acc, iid_count)
        ood_ci = calculate_confidence_interval(ood_acc, ood_count)

        results[label] = {
            "iid_accuracy": iid_acc,
            "ood_accuracy": ood_acc,
            "iid_ci": iid_ci,
            "ood_ci": ood_ci,
            "iid_count": iid_count,
            "ood_count": ood_count,
        }

        print(f"{label}:")
        print(f"  IID Accuracy: {iid_acc:.2%} ± {iid_ci:.2%} (n={iid_count})")
        print(f"  OOD Accuracy: {ood_acc:.2%} ± {ood_ci:.2%} (n={ood_count})")

    return results


def plot_accuracies(results, output_path=None):
    """Create bar plots for IID and OOD accuracies."""
    labels = list(results.keys())
    iid_accs = [results[name]["iid_accuracy"] for name in labels]
    ood_accs = [results[name]["ood_accuracy"] for name in labels]
    iid_cis = [results[name]["iid_ci"] for name in labels]
    ood_cis = [results[name]["ood_ci"] for name in labels]

    # Generate distinct colors for each bar
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # IID plot
    x_pos = np.arange(len(labels))
    bars1 = ax1.bar(x_pos, iid_accs, color=colors, alpha=0.8, yerr=iid_cis, capsize=5)
    ax1.set_xlabel("LoRA Adapter", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax1.set_title("IID Dataset Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.set_ylim([0, 1])
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.1%}", ha="center", va="bottom", fontsize=9)

    # OOD plot
    bars2 = ax2.bar(x_pos, ood_accs, color=colors, alpha=0.8, yerr=ood_cis, capsize=5)
    ax2.set_xlabel("LoRA Adapter", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax2.set_title("OOD Dataset Accuracy", fontsize=14, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha="right")
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


def main():
    print(f"Loading results from: {INPUT_FOLDER}\n")
    results = load_results_from_folder(INPUT_FOLDER)

    if not results:
        print("No JSON files found in the specified folder!")
        return

    print(f"\nGenerating plots...")
    plot_accuracies(results, OUTPUT_PATH)


if __name__ == "__main__":
    main()
