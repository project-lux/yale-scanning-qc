"""
Prepare augmented data for YOLO26 training.
Converts augmented data into YOLO-compatible format.
"""

import json
import random
import shutil
from pathlib import Path
from typing import Optional

import cv2
import yaml
from tqdm import tqdm

from ..config import AUGMENTED_DIR, DATA_DIR, QC_CATEGORIES, RANDOM_SEED


class DatasetPreparer:
    """
    Prepares datasets in YOLO format for training.

    Creates the following structure:
    datasets/
      {task_name}/
        train/
          images/
          labels/
        val/
          images/
          labels/
        data.yaml
    """

    def __init__(
        self,
        augmented_dir: Path = AUGMENTED_DIR,
        output_dir: Optional[Path] = None,
        seed: int = RANDOM_SEED,
    ):
        self.augmented_dir = augmented_dir
        self.output_dir = output_dir or DATA_DIR / "datasets"
        self.seed = seed
        random.seed(seed)

    def prepare_classification_dataset(
        self,
        category: str,
        classes: list[str],
        train_ratio: float = 0.8,
    ) -> Path:
        """
        Prepare a classification dataset (focus, orientation, dust).

        YOLO classification expects:
        train/
          class1/
            img1.jpg
          class2/
            img2.jpg
        val/
          ...
        """
        source_dir = self.augmented_dir / category
        dataset_dir = self.output_dir / f"{category}_cls"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Create class directories
        for split in ["train", "val"]:
            for cls in classes:
                (dataset_dir / split / cls).mkdir(parents=True, exist_ok=True)

        # Collect all images with their labels
        images_by_class = {cls: [] for cls in classes}

        image_files = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
        print(f"Found {len(image_files)} images for {category}")

        for img_path in image_files:
            # Look for corresponding JSON
            json_path = img_path.with_suffix(".json")
            if json_path.exists():
                with open(json_path) as f:
                    meta = json.load(f)
                cls = meta.get("class")
                if cls in classes:
                    images_by_class[cls].append(img_path)

        # Split and copy
        for cls, images in images_by_class.items():
            random.shuffle(images)
            split_idx = int(len(images) * train_ratio)

            for i, img_path in enumerate(images):
                split = "train" if i < split_idx else "val"
                dst = dataset_dir / split / cls / img_path.name
                shutil.copy2(img_path, dst)

            print(f"  {cls}: {split_idx} train, {len(images) - split_idx} val")

        # Create data.yaml
        data_yaml = {
            "path": str(dataset_dir.absolute()),
            "train": "train",
            "val": "val",
            "nc": len(classes),
            "names": classes,
        }

        yaml_path = dataset_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        print(f"Created classification dataset at {dataset_dir}")
        return dataset_dir

    def prepare_detection_dataset(
        self,
        category: str,
        classes: list[str],
        train_ratio: float = 0.8,
    ) -> Path:
        """
        Prepare a detection dataset (obstruction).

        YOLO detection expects:
        train/
          images/
            img1.jpg
          labels/
            img1.txt  (class_id cx cy w h - normalized)
        val/
          ...
        """
        source_dir = self.augmented_dir / category
        dataset_dir = self.output_dir / f"{category}_det"

        for split in ["train", "val"]:
            (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        # Collect all images
        image_files = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
        random.shuffle(image_files)

        split_idx = int(len(image_files) * train_ratio)
        print(f"Found {len(image_files)} images for {category}")

        for i, img_path in enumerate(tqdm(image_files, desc=f"Preparing {category}")):
            split = "train" if i < split_idx else "val"

            # Copy image
            dst_img = dataset_dir / split / "images" / img_path.name
            shutil.copy2(img_path, dst_img)

            # Convert annotations to YOLO format
            json_path = img_path.with_suffix(".json")
            txt_path = dataset_dir / split / "labels" / img_path.with_suffix(".txt").name

            if json_path.exists():
                with open(json_path) as f:
                    meta = json.load(f)

                annotations = meta.get("annotations", [])
                with open(txt_path, "w") as f:
                    for ann in annotations:
                        if "bbox_normalized" in ann:
                            # bbox_normalized is [x, y, w, h] in 0-1 range
                            x, y, w, h = ann["bbox_normalized"]
                            # YOLO wants center x, center y, width, height
                            cx = x + w / 2
                            cy = y + h / 2
                            class_id = ann.get("class_id", 0)
                            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            else:
                # Empty label file (no detections)
                txt_path.touch()

        # Create data.yaml
        data_yaml = {
            "path": str(dataset_dir.absolute()),
            "train": "train/images",
            "val": "val/images",
            "nc": len(classes),
            "names": classes,
        }

        yaml_path = dataset_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        print(f"Created detection dataset at {dataset_dir}")
        return dataset_dir

    def prepare_all_datasets(self) -> dict[str, Path]:
        """Prepare datasets for all QC categories."""
        datasets = {}

        for category, config in QC_CATEGORIES.items():
            if config.get("type") == "algorithmic":
                print(f"Skipping {category} (algorithmic, no training needed)")
                continue

            classes = config.get("classes", [])
            task_type = config.get("type")

            print(f"\nPreparing {category} dataset ({task_type})...")

            if task_type == "classification":
                dataset_path = self.prepare_classification_dataset(category, classes)
            elif task_type == "detection":
                dataset_path = self.prepare_detection_dataset(category, classes)
            else:
                print(f"Unknown task type: {task_type}")
                continue

            datasets[category] = dataset_path

        return datasets


def main():
    """CLI entry point for dataset preparation."""
    import argparse

    parser = argparse.ArgumentParser(description="Prepare datasets for YOLO training")
    parser.add_argument(
        "-i", "--input-dir",
        type=Path,
        default=AUGMENTED_DIR,
        help=f"Augmented data directory (default: {AUGMENTED_DIR})"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory for datasets"
    )
    parser.add_argument(
        "-c", "--category",
        choices=list(QC_CATEGORIES.keys()),
        help="Specific category to prepare (default: all)"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})"
    )

    args = parser.parse_args()

    preparer = DatasetPreparer(
        augmented_dir=args.input_dir,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    if args.category:
        config = QC_CATEGORIES[args.category]
        if config.get("type") == "classification":
            preparer.prepare_classification_dataset(args.category, config["classes"])
        elif config.get("type") == "detection":
            preparer.prepare_detection_dataset(args.category, config["classes"])
    else:
        preparer.prepare_all_datasets()


if __name__ == "__main__":
    main()

