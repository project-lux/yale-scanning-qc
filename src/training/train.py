"""
YOLO26 training workflow for QC models.
"""

from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from ..config import (
    BATCH_SIZE,
    DATA_DIR,
    EPOCHS,
    IMG_SIZE,
    MODELS_DIR,
    QC_CATEGORIES,
    YOLO_MODEL_SIZE,
)


class YOLOTrainer:
    """
    Trainer for YOLO26 models.

    Supports:
    - Classification (focus, orientation, dust)
    - Object detection (obstruction)
    """

    def __init__(
        self,
        model_size: str = YOLO_MODEL_SIZE,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        img_size: int = IMG_SIZE,
        output_dir: Path = MODELS_DIR,
    ):
        self.model_size = model_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_classification(
        self,
        data_path: Path,
        name: str,
        pretrained: Optional[str] = None,
    ) -> Path:
        """
        Train a classification model.

        Args:
            data_path: Path to dataset directory (with train/val subdirs)
            name: Model name for saving
            pretrained: Optional pretrained weights path

        Returns:
            Path to best trained weights
        """
        # Use YOLO26 classification model
        if pretrained:
            model = YOLO(pretrained)
        else:
            model = YOLO(f"yolo26{self.model_size}-cls.pt")

        print(f"\nTraining classification model: {name}")
        print(f"  Dataset: {data_path}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Image size: {self.img_size}")

        # Train
        results = model.train(
            data=str(data_path),
            epochs=self.epochs,
            batch=self.batch_size,
            imgsz=self.img_size,
            project=str(self.output_dir),
            name=name,
            exist_ok=True,
            pretrained=True,
            verbose=True,
        )

        # Copy best weights to standard location
        best_weights = Path(results.save_dir) / "weights" / "best.pt"
        output_path = self.output_dir / f"yolo26-{name}.pt"

        if best_weights.exists():
            import shutil
            shutil.copy2(best_weights, output_path)
            print(f"Saved model to {output_path}")
            return output_path

        return best_weights

    def train_detection(
        self,
        data_yaml: Path,
        name: str,
        pretrained: Optional[str] = None,
    ) -> Path:
        """
        Train a detection model.

        Args:
            data_yaml: Path to data.yaml file
            name: Model name for saving
            pretrained: Optional pretrained weights path

        Returns:
            Path to best trained weights
        """
        # Use YOLO26 detection model
        if pretrained:
            model = YOLO(pretrained)
        else:
            model = YOLO(f"yolo26{self.model_size}.pt")

        print(f"\nTraining detection model: {name}")
        print(f"  Dataset: {data_yaml}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Image size: {self.img_size}")

        # Train
        results = model.train(
            data=str(data_yaml),
            epochs=self.epochs,
            batch=self.batch_size,
            imgsz=self.img_size,
            project=str(self.output_dir),
            name=name,
            exist_ok=True,
            pretrained=True,
            verbose=True,
        )

        # Copy best weights to standard location
        best_weights = Path(results.save_dir) / "weights" / "best.pt"
        output_path = self.output_dir / f"yolo26-{name}.pt"

        if best_weights.exists():
            import shutil
            shutil.copy2(best_weights, output_path)
            print(f"Saved model to {output_path}")
            return output_path

        return best_weights

    def train_category(self, category: str, dataset_dir: Optional[Path] = None) -> Path:
        """
        Train model for a specific QC category.

        Args:
            category: Category name (focus, obstruction, orientation, dust)
            dataset_dir: Optional custom dataset directory

        Returns:
            Path to trained model
        """
        config = QC_CATEGORIES.get(category)
        if not config:
            raise ValueError(f"Unknown category: {category}")

        if config.get("type") == "algorithmic":
            raise ValueError(f"{category} uses algorithmic approach, no training needed")

        if dataset_dir is None:
            task_suffix = "cls" if config["type"] == "classification" else "det"
            dataset_dir = DATA_DIR / "datasets" / f"{category}_{task_suffix}"

        if not dataset_dir.exists():
            raise ValueError(f"Dataset not found: {dataset_dir}")

        if config["type"] == "classification":
            return self.train_classification(dataset_dir, category)
        elif config["type"] == "detection":
            data_yaml = dataset_dir / "data.yaml"
            return self.train_detection(data_yaml, category)
        else:
            raise ValueError(f"Unknown task type: {config['type']}")

    def train_all(self) -> dict[str, Path]:
        """Train models for all trainable categories."""
        results = {}

        for category, config in QC_CATEGORIES.items():
            if config.get("type") == "algorithmic":
                print(f"\nSkipping {category} (algorithmic)")
                continue

            try:
                model_path = self.train_category(category)
                results[category] = model_path
            except Exception as e:
                print(f"Error training {category}: {e}")
                results[category] = None

        return results


def main():
    """CLI entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLO26 QC models")
    parser.add_argument(
        "-c", "--category",
        choices=[c for c, cfg in QC_CATEGORIES.items() if cfg.get("type") != "algorithmic"],
        help="Category to train (default: all)"
    )
    parser.add_argument(
        "-d", "--dataset-dir",
        type=Path,
        help="Custom dataset directory"
    )
    parser.add_argument(
        "-s", "--model-size",
        choices=["n", "s", "m", "l", "x"],
        default=YOLO_MODEL_SIZE,
        help=f"YOLO26 model size (default: {YOLO_MODEL_SIZE})"
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Training epochs (default: {EPOCHS})"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "-i", "--img-size",
        type=int,
        default=IMG_SIZE,
        help=f"Image size (default: {IMG_SIZE})"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=MODELS_DIR,
        help=f"Output directory for models (default: {MODELS_DIR})"
    )

    args = parser.parse_args()

    trainer = YOLOTrainer(
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        output_dir=args.output_dir,
    )

    if args.category:
        trainer.train_category(args.category, args.dataset_dir)
    else:
        trainer.train_all()


if __name__ == "__main__":
    main()

