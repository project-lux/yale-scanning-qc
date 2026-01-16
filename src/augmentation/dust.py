"""
Dust augmentation for dust/debris object detection training.
Adds realistic dust, scratches, and debris artifacts with bounding box annotations.
"""

import random
from typing import Optional

import cv2
import numpy as np


class DustAugmenter:
    """
    Augmenter that adds dust, debris, and scratch artifacts to images.
    Returns bounding box annotations for object detection training.

    Artifact types (classes):
    - dust_speck (class_id: 0): Small dark or light spots
    - scratch (class_id: 1): Linear marks
    - smudge (class_id: 2): Larger blurred areas
    - fiber (class_id: 3): Hair/fiber artifacts
    """

    CLASS_NAMES = ["dust_speck", "scratch", "smudge", "fiber"]
    CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        np.random.seed(seed)

    def add_dust_specks(
        self,
        image: np.ndarray,
        num_specks: Optional[int] = None,
        speck_size_range: tuple[int, int] = (3, 12),
    ) -> tuple[np.ndarray, list[dict]]:
        """Add small dust specks to the image with bounding boxes."""
        result = image.copy()
        h, w = image.shape[:2]
        annotations = []

        if num_specks is None:
            num_specks = self.rng.randint(5, 20)

        for _ in range(num_specks):
            x = self.rng.randint(10, w - 10)
            y = self.rng.randint(10, h - 10)
            size = self.rng.randint(*speck_size_range)

            # Random brightness (dark or light specks)
            if self.rng.random() > 0.3:
                color = tuple(self.rng.randint(0, 50) for _ in range(3))
            else:
                color = tuple(self.rng.randint(200, 255) for _ in range(3))

            # Draw speck
            overlay = result.copy()
            cv2.circle(overlay, (x, y), size, color, -1)
            alpha = self.rng.uniform(0.5, 0.9)
            result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)

            # Create bounding box (convert to native Python int)
            bbox = [
                int(max(0, x - size)),
                int(max(0, y - size)),
                int(min(w, x + size) - max(0, x - size)),
                int(min(h, y + size) - max(0, y - size)),
            ]

            if bbox[2] > 2 and bbox[3] > 2:  # Only add if visible
                annotations.append({
                    "class": "dust_speck",
                    "class_id": 0,
                    "bbox": bbox,
                    "bbox_normalized": [
                        float(bbox[0] / w),
                        float(bbox[1] / h),
                        float(bbox[2] / w),
                        float(bbox[3] / h),
                    ],
                })

        return result, annotations

    def add_scratches(
        self,
        image: np.ndarray,
        num_scratches: Optional[int] = None,
    ) -> tuple[np.ndarray, list[dict]]:
        """Add linear scratch marks with bounding boxes."""
        result = image.copy()
        h, w = image.shape[:2]
        annotations = []

        if num_scratches is None:
            num_scratches = self.rng.randint(1, 4)

        for _ in range(num_scratches):
            x1 = self.rng.randint(0, w)
            y1 = self.rng.randint(0, h)

            angle = self.rng.uniform(0, 360)
            length = self.rng.randint(50, min(w, h) // 3)

            x2 = int(x1 + length * np.cos(np.radians(angle)))
            y2 = int(y1 + length * np.sin(np.radians(angle)))

            color = tuple(self.rng.randint(180, 255) for _ in range(3))
            thickness = self.rng.randint(1, 3)

            overlay = result.copy()
            cv2.line(overlay, (x1, y1), (x2, y2), color, thickness)

            alpha = self.rng.uniform(0.3, 0.6)
            result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)

            # Calculate bounding box (convert to native Python int)
            x_min = int(max(0, min(x1, x2) - thickness))
            y_min = int(max(0, min(y1, y2) - thickness))
            x_max = int(min(w, max(x1, x2) + thickness))
            y_max = int(min(h, max(y1, y2) + thickness))

            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

            if bbox[2] > 5 and bbox[3] > 5:
                annotations.append({
                    "class": "scratch",
                    "class_id": 1,
                    "bbox": bbox,
                    "bbox_normalized": [
                        float(bbox[0] / w),
                        float(bbox[1] / h),
                        float(bbox[2] / w),
                        float(bbox[3] / h),
                    ],
                })

        return result, annotations

    def add_smudges(
        self,
        image: np.ndarray,
        num_smudges: Optional[int] = None,
    ) -> tuple[np.ndarray, list[dict]]:
        """Add smudge/fingerprint-like artifacts with bounding boxes."""
        result = image.copy()
        h, w = image.shape[:2]
        annotations = []

        if num_smudges is None:
            num_smudges = self.rng.randint(1, 3)

        for _ in range(num_smudges):
            center_x = self.rng.randint(50, w - 50)
            center_y = self.rng.randint(50, h - 50)
            axis_x = self.rng.randint(30, 100)
            axis_y = self.rng.randint(20, 60)
            angle = self.rng.uniform(0, 360)

            mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(mask, (center_x, center_y), (axis_x, axis_y), angle, 0, 360, 1.0, -1)

            blur_size = self.rng.choice([21, 31, 41])
            mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 10)

            if self.rng.random() > 0.5:
                factor = self.rng.uniform(0.7, 0.9)
                result = (result * (1 - mask[:, :, np.newaxis] * (1 - factor))).astype(np.uint8)
            else:
                factor = self.rng.uniform(0.1, 0.3)
                result = (result + mask[:, :, np.newaxis] * 255 * factor).clip(0, 255).astype(np.uint8)

            # Calculate bounding box from ellipse (convert to native Python int)
            cos_a = np.cos(np.radians(angle))
            sin_a = np.sin(np.radians(angle))
            half_w = np.sqrt((axis_x * cos_a) ** 2 + (axis_y * sin_a) ** 2)
            half_h = np.sqrt((axis_x * sin_a) ** 2 + (axis_y * cos_a) ** 2)

            x_min = int(max(0, int(center_x - half_w)))
            y_min = int(max(0, int(center_y - half_h)))
            x_max = int(min(w, int(center_x + half_w)))
            y_max = int(min(h, int(center_y + half_h)))

            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

            if bbox[2] > 10 and bbox[3] > 10:
                annotations.append({
                    "class": "smudge",
                    "class_id": 2,
                    "bbox": bbox,
                    "bbox_normalized": [
                        float(bbox[0] / w),
                        float(bbox[1] / h),
                        float(bbox[2] / w),
                        float(bbox[3] / h),
                    ],
                })

        return result, annotations

    def add_fibers(
        self,
        image: np.ndarray,
        num_fibers: Optional[int] = None,
    ) -> tuple[np.ndarray, list[dict]]:
        """Add hair/fiber artifacts with bounding boxes."""
        result = image.copy()
        h, w = image.shape[:2]
        annotations = []

        if num_fibers is None:
            num_fibers = self.rng.randint(1, 3)

        for _ in range(num_fibers):
            num_points = self.rng.randint(4, 8)
            points = []

            x = self.rng.randint(20, w - 20)
            y = self.rng.randint(20, h - 20)
            points.append([x, y])

            for _ in range(num_points - 1):
                dx = self.rng.randint(-40, 40)
                dy = self.rng.randint(-25, 25)
                x = max(0, min(w - 1, x + dx))
                y = max(0, min(h - 1, y + dy))
                points.append([x, y])

            points = np.array(points, dtype=np.int32)

            color = tuple(self.rng.randint(10, 60) for _ in range(3))
            thickness = self.rng.randint(1, 2)

            overlay = result.copy()
            cv2.polylines(overlay, [points], False, color, thickness)

            alpha = self.rng.uniform(0.5, 0.8)
            result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)

            # Calculate bounding box (convert to native Python int)
            x_min = int(max(0, points[:, 0].min() - 5))
            y_min = int(max(0, points[:, 1].min() - 5))
            x_max = int(min(w, points[:, 0].max() + 5))
            y_max = int(min(h, points[:, 1].max() + 5))

            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

            if bbox[2] > 5 and bbox[3] > 5:
                annotations.append({
                    "class": "fiber",
                    "class_id": 3,
                    "bbox": bbox,
                    "bbox_normalized": [
                        float(bbox[0] / w),
                        float(bbox[1] / h),
                        float(bbox[2] / w),
                        float(bbox[3] / h),
                    ],
                })

        return result, annotations

    def augment(
        self,
        image: np.ndarray,
        severity: str = "medium",
        artifact_types: Optional[list[str]] = None,
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Apply dust/debris augmentation with bounding box annotations.

        Args:
            image: Input image
            severity: Amount of artifacts (light, medium, heavy)
            artifact_types: List of artifact types to apply, or None for random

        Returns:
            Tuple of (augmented image, list of annotation dicts with bboxes)
        """
        result = image.copy()
        all_annotations = []

        severity_config = {
            "light": {"specks": (3, 8), "scratches": (0, 1), "smudges": (0, 1), "fibers": (0, 1)},
            "medium": {"specks": (5, 12), "scratches": (1, 2), "smudges": (1, 2), "fibers": (1, 2)},
            "heavy": {"specks": (10, 20), "scratches": (2, 4), "smudges": (2, 3), "fibers": (2, 4)},
        }
        config = severity_config.get(severity, severity_config["medium"])

        if artifact_types is None:
            artifact_types = []
            if self.rng.random() > 0.3:
                artifact_types.append("specks")
            if self.rng.random() > 0.4:
                artifact_types.append("scratches")
            if self.rng.random() > 0.5:
                artifact_types.append("smudges")
            if self.rng.random() > 0.4:
                artifact_types.append("fibers")

            if not artifact_types:
                artifact_types = ["specks"]

        for artifact_type in artifact_types:
            if artifact_type == "specks":
                num = self.rng.randint(*config["specks"])
                result, annotations = self.add_dust_specks(result, num)
                all_annotations.extend(annotations)
            elif artifact_type == "scratches":
                num = self.rng.randint(*config["scratches"])
                if num > 0:
                    result, annotations = self.add_scratches(result, num)
                    all_annotations.extend(annotations)
            elif artifact_type == "smudges":
                num = self.rng.randint(*config["smudges"])
                if num > 0:
                    result, annotations = self.add_smudges(result, num)
                    all_annotations.extend(annotations)
            elif artifact_type == "fibers":
                num = self.rng.randint(*config["fibers"])
                if num > 0:
                    result, annotations = self.add_fibers(result, num)
                    all_annotations.extend(annotations)

        return result, all_annotations

    def create_clean_sample(self, image: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """Return original clean image with no annotations (negative sample)."""
        return image.copy(), []
