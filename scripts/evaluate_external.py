import csv
import json
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import IMAGE_SIZE, MODEL_DIR
from src.model import BasicCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(
    r"C:\Users\mda16\Downloads\VerityAI_Evaluation\sample"
)
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_PATH = MODEL_DIR / "best_basic_cnn.pth"

# Dataset mapping: 0 = fake, 1 = real.
CLASS_TO_LABEL = {"fake": 0, "real": 1}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class ExternalFaceDataset(Dataset):
    def __init__(self, root):
        self.samples = []

        for class_name, label in CLASS_TO_LABEL.items():
            class_dir = root / class_name

            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing class directory: {class_dir}"
                )

            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in {
                    ".jpg", ".jpeg", ".png", ".webp"
                }:
                    self.samples.append((path, label))

        if not self.samples:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = preprocess(image)

        return tensor, label, str(path)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model = BasicCNN(num_classes=2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    dataset = ExternalFaceDataset(DATASET_ROOT)

    counts = {
        "fake": sum(label == 0 for _, label in dataset.samples),
        "real": sum(label == 1 for _, label in dataset.samples),
    }

    if counts != {"fake": 500, "real": 500}:
        raise ValueError(
            f"Expected 500 images per class, found: {counts}"
        )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    y_true = []
    y_pred = []
    rows = []

    with torch.inference_mode():
        for images, labels, paths in loader:
            images = images.to(DEVICE)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1).cpu().tolist()
            labels = labels.tolist()
            probabilities = probabilities.cpu().tolist()

            for path, true_label, pred_label, probs in zip(
                paths, labels, predictions, probabilities
            ):
                y_true.append(true_label)
                y_pred.append(pred_label)

                rows.append({
                    "path": path,
                    "true_label": "REAL" if true_label == 1 else "AI_GENERATED",
                    "predicted_label": (
                        "REAL" if pred_label == 1 else "AI_GENERATED"
                    ),
                    "prob_fake": probs[0],
                    "prob_real": probs[1],
                    "confidence": max(probs),
                    "correct": true_label == pred_label,
                })

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        average="macro",
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["AI_GENERATED", "REAL"],
        output_dict=True,
        zero_division=0,
    )

    summary = {
        "experiment": "External face dataset evaluation",
        "dataset": "DeepFakeFace sample",
        "num_images": len(dataset),
        "class_counts": {
            "AI_GENERATED": counts["fake"],
            "REAL": counts["real"],
        },
        "model": "BasicCNN",
        "checkpoint": str(CHECKPOINT_PATH),
        "image_size": IMAGE_SIZE,
        "device": str(DEVICE),
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "confusion_matrix_labels": ["AI_GENERATED", "REAL"],
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "note": (
            "Results apply to this sampled dataset only. "
            "They do not establish general real-world accuracy."
        ),
    }

    json_path = RESULTS_DIR / "external_evaluation_results.json"
    csv_path = RESULTS_DIR / "external_evaluation_predictions.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== EXTERNAL EVALUATION RESULTS =====")
    print(f"Device: {DEVICE}")
    print(f"Images: {len(dataset)}")
    print(f"Class counts: {counts}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro precision: {precision:.4f}")
    print(f"Macro recall: {recall:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print("Labels: [AI_GENERATED, REAL]")
    print(cm)
    print(f"\nSaved metrics: {json_path}")
    print(f"Saved predictions: {csv_path}")


if __name__ == "__main__":
    main()