"""
Main augmentation pipeline that orchestrates all augmenters.
"""

import json
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from ..config import (
    AUGMENTATION_MULTIPLIER,
    AUGMENTED_DIR,
    QC_CATEGORIES,
    RANDOM_SEED,
    RAW_IMAGES_DIR,
)
from .blur import BlurAugmenter
from .dust import DustAugmenter
from .obstruction import ObstructionAugmenter
from .orientation import OrientationAugmenter
from .skew import SkewAugmenter


class AugmentationPipeline:
    """
    Orchestrates all augmentation modules to create training datasets.
    """

    def __init__(self, seed: int = RANDOM_SEED):
        self.seed = seed
        self.rng = random.Random(seed)

        # Initialize augmenters
        # Note: Black background is applied during download, not augmentation
        self.blur_aug = BlurAugmenter(seed)
        self.obstruction_aug = ObstructionAugmenter(seed)
        self.orientation_aug = OrientationAugmenter(seed)
        self.dust_aug = DustAugmenter(seed)
        self.skew_aug = SkewAugmenter(seed)

    def augment_for_focus(
        self,
        image: np.ndarray,
        num_samples: int = 4,
    ) -> list[tuple[np.ndarray, dict]]:
        """Generate focus training samples (in_focus vs out_of_focus)."""
        samples = []

        # Create sharp (in_focus) samples
        for _ in range(num_samples // 2):
            sample, meta = self.blur_aug.create_sharp_sample(image)
            samples.append((sample, meta))

        # Create blurred (out_of_focus) samples with varying severity
        severities = ["light", "medium", "heavy"]
        blur_types = ["gaussian", "motion", "defocus", "partial"]

        for _ in range(num_samples - num_samples // 2):
            severity = self.rng.choice(severities)
            blur_type = self.rng.choice(blur_types)
            sample, meta = self.blur_aug.augment(image, blur_type, severity)
            samples.append((sample, meta))

        return samples

    def augment_for_obstruction(
        self,
        image: np.ndarray,
        num_samples: int = 4,
    ) -> list[tuple[np.ndarray, list[dict]]]:
        """Generate obstruction detection samples with bboxes."""
        samples = []

        # Image already has black background from download
        # Clean samples (no obstruction)
        for _ in range(num_samples // 4):
            sample, annotations = self.obstruction_aug.create_clean_sample(image)
            samples.append((sample, annotations))

        # Obstructed samples - obstructions appear on top of the scanned image
        obstruction_types = ["hand", "finger", "shadow", "object"]
        for _ in range(num_samples - num_samples // 4):
            obs_type = self.rng.choice(obstruction_types)
            sample, annotations = self.obstruction_aug.augment(image, obs_type)
            samples.append((sample, annotations))

        return samples

    def augment_for_orientation(
        self,
        image: np.ndarray,
        num_samples: int = 4,
    ) -> list[tuple[np.ndarray, dict]]:
        """Generate orientation classification samples with clean 90° rotations only."""
        # Image already has black background from download

        # Create balanced samples (one per class: correct, 90cw, 180, 90ccw)
        samples = self.orientation_aug.create_balanced_samples(image)

        # Add more if needed (randomly pick from the 4 classes)
        while len(samples) < num_samples:
            target_class = self.rng.choice(OrientationAugmenter.CLASSES)
            sample, meta = self.orientation_aug.augment(image, target_class)
            samples.append((sample, meta))

        return samples[:num_samples]

    def augment_for_dust(
        self,
        image: np.ndarray,
        num_samples: int = 4,
    ) -> list[tuple[np.ndarray, list[dict]]]:
        """Generate dust detection samples with bounding box annotations."""
        samples = []

        # Image already has black background from download
        # Clean samples (no dust = no annotations)
        for _ in range(num_samples // 4):
            sample, annotations = self.dust_aug.create_clean_sample(image)
            samples.append((sample, annotations))

        # Dusty samples - dust appears on the scanner/image
        severities = ["light", "medium", "heavy"]
        for _ in range(num_samples - num_samples // 4):
            severity = self.rng.choice(severities)
            sample, annotations = self.dust_aug.augment(image, severity)
            samples.append((sample, annotations))

        return samples

    def augment_for_skew(
        self,
        image: np.ndarray,
        num_samples: int = 4,
    ) -> list[tuple[np.ndarray, dict]]:
        """Generate skew samples for visual reference (not for ML training)."""
        samples = []

        # Image already has black background from download
        # Aligned samples
        for _ in range(num_samples // 2):
            sample, meta = self.skew_aug.create_aligned_sample(image)
            samples.append((sample, meta))

        # Skewed samples
        severities = ["light", "medium", "heavy"]
        skew_types = ["perspective", "rotation", "affine"]
        for _ in range(num_samples - num_samples // 2):
            severity = self.rng.choice(severities)
            skew_type = self.rng.choice(skew_types)
            sample, meta = self.skew_aug.augment(image, skew_type, severity)
            samples.append((sample, meta))

        return samples

    def process_image(
        self,
        image_path: Path,
        output_dir: Path,
        categories: Optional[list[str]] = None,
        num_samples_per_category: int = AUGMENTATION_MULTIPLIER,
    ) -> dict:
        """
        Process a single image through all augmentation pipelines.

        Args:
            image_path: Path to input image
            output_dir: Base output directory
            categories: List of categories to augment for, or None for all
            num_samples_per_category: Number of samples per category

        Returns:
            Dictionary with statistics
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return {"error": f"Could not load {image_path}"}

        if categories is None:
            categories = list(QC_CATEGORIES.keys())

        stats = {"image": image_path.name, "samples": {}}
        base_name = image_path.stem

        for category in categories:
            cat_dir = output_dir / category
            cat_dir.mkdir(parents=True, exist_ok=True)

            if category == "focus":
                samples = self.augment_for_focus(image, num_samples_per_category)
                for i, (sample, meta) in enumerate(samples):
                    filename = f"{base_name}_focus_{i}.jpg"
                    cv2.imwrite(str(cat_dir / filename), sample)
                    # Save metadata
                    with open(cat_dir / f"{base_name}_focus_{i}.json", "w") as f:
                        json.dump(meta, f)
                stats["samples"]["focus"] = len(samples)

            elif category == "obstruction":
                samples = self.augment_for_obstruction(image, num_samples_per_category)
                for i, (sample, annotations) in enumerate(samples):
                    filename = f"{base_name}_obs_{i}.jpg"
                    cv2.imwrite(str(cat_dir / filename), sample)
                    # Save YOLO-format annotations
                    with open(cat_dir / f"{base_name}_obs_{i}.json", "w") as f:
                        json.dump({"annotations": annotations}, f)
                stats["samples"]["obstruction"] = len(samples)

            elif category == "orientation":
                samples = self.augment_for_orientation(image, num_samples_per_category)
                for i, (sample, meta) in enumerate(samples):
                    filename = f"{base_name}_orient_{i}.jpg"
                    cv2.imwrite(str(cat_dir / filename), sample)
                    with open(cat_dir / f"{base_name}_orient_{i}.json", "w") as f:
                        json.dump(meta, f)
                stats["samples"]["orientation"] = len(samples)

            elif category == "dust":
                samples = self.augment_for_dust(image, num_samples_per_category)
                for i, (sample, annotations) in enumerate(samples):
                    filename = f"{base_name}_dust_{i}.jpg"
                    cv2.imwrite(str(cat_dir / filename), sample)
                    # Save YOLO-format annotations for object detection
                    with open(cat_dir / f"{base_name}_dust_{i}.json", "w") as f:
                        json.dump({"annotations": annotations}, f)
                stats["samples"]["dust"] = len(samples)

            elif category == "skew":
                samples = self.augment_for_skew(image, num_samples_per_category)
                for i, (sample, meta) in enumerate(samples):
                    filename = f"{base_name}_skew_{i}.jpg"
                    cv2.imwrite(str(cat_dir / filename), sample)
                    with open(cat_dir / f"{base_name}_skew_{i}.json", "w") as f:
                        json.dump(meta, f)
                stats["samples"]["skew"] = len(samples)

        return stats

    def process_directory(
        self,
        input_dir: Path = RAW_IMAGES_DIR,
        output_dir: Path = AUGMENTED_DIR,
        categories: Optional[list[str]] = None,
        num_samples_per_category: int = AUGMENTATION_MULTIPLIER,
    ) -> dict:
        """
        Process all images in a directory.

        Returns:
            Dictionary with overall statistics
        """
        image_paths = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
        print(f"Found {len(image_paths)} images to process")

        total_samples = {cat: 0 for cat in (categories or list(QC_CATEGORIES.keys()))}
        errors = []

        for image_path in tqdm(image_paths, desc="Augmenting images"):
            result = self.process_image(
                image_path, output_dir, categories, num_samples_per_category
            )

            if "error" in result:
                errors.append(result["error"])
            else:
                for cat, count in result.get("samples", {}).items():
                    total_samples[cat] += count

        return {
            "images_processed": len(image_paths) - len(errors),
            "errors": len(errors),
            "samples_created": total_samples,
        }


def main():
    """CLI entry point for augmentation pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Augment images for QC training")
    parser.add_argument(
        "-i", "--input-dir",
        type=Path,
        default=RAW_IMAGES_DIR,
        help=f"Input directory (default: {RAW_IMAGES_DIR})"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=AUGMENTED_DIR,
        help=f"Output directory (default: {AUGMENTED_DIR})"
    )
    parser.add_argument(
        "-c", "--categories",
        nargs="+",
        choices=list(QC_CATEGORIES.keys()),
        help="Categories to augment (default: all)"
    )
    parser.add_argument(
        "-n", "--num-samples",
        type=int,
        default=AUGMENTATION_MULTIPLIER,
        help=f"Samples per category per image (default: {AUGMENTATION_MULTIPLIER})"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})"
    )

    args = parser.parse_args()

    pipeline = AugmentationPipeline(seed=args.seed)
    stats = pipeline.process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        categories=args.categories,
        num_samples_per_category=args.num_samples,
    )

    print("\nAugmentation complete!")
    print(f"Images processed: {stats['images_processed']}")
    print(f"Errors: {stats['errors']}")
    print("Samples created:")
    for cat, count in stats["samples_created"].items():
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()

