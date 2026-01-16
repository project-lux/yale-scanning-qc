"""
Orientation augmentation for rotation detection training.
Creates rotated versions of images for 4-class orientation classification.
"""

import random
from typing import Optional

import cv2
import numpy as np


class OrientationAugmenter:
    """
    Augmenter for creating orientation training samples.

    Classes:
    - correct: No rotation needed
    - rotate_90_cw: Image needs 90° clockwise rotation to be correct
    - rotate_180: Image needs 180° rotation to be correct
    - rotate_90_ccw: Image needs 90° counter-clockwise rotation to be correct

    Note: We apply the INVERSE rotation to create training data.
    E.g., to create a "rotate_90_cw" sample (image that needs CW rotation),
    we rotate the original CCW by 90°.
    """

    CLASSES = ["correct", "rotate_90_cw", "rotate_180", "rotate_90_ccw"]
    CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def rotate_90_ccw(self, image: np.ndarray) -> np.ndarray:
        """Rotate image 90° counter-clockwise."""
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def rotate_90_cw(self, image: np.ndarray) -> np.ndarray:
        """Rotate image 90° clockwise."""
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    def rotate_180(self, image: np.ndarray) -> np.ndarray:
        """Rotate image 180°."""
        return cv2.rotate(image, cv2.ROTATE_180)

    def augment(
        self,
        image: np.ndarray,
        target_class: Optional[str] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Create orientation training sample.

        Args:
            image: Input image (assumed to be correctly oriented)
            target_class: Target orientation class (or None for random)

        Returns:
            Tuple of (rotated image, metadata dict with class info)
        """
        if target_class is None:
            target_class = self.rng.choice(self.CLASSES)

        if target_class not in self.CLASSES:
            raise ValueError(f"Invalid class: {target_class}. Must be one of {self.CLASSES}")

        # To create samples that need a specific correction,
        # we apply the inverse transformation
        if target_class == "correct":
            result = image.copy()
        elif target_class == "rotate_90_cw":
            # Image needs CW rotation, so we rotated it CCW
            result = self.rotate_90_ccw(image)
        elif target_class == "rotate_180":
            result = self.rotate_180(image)
        elif target_class == "rotate_90_ccw":
            # Image needs CCW rotation, so we rotated it CW
            result = self.rotate_90_cw(image)
        else:
            result = image.copy()

        metadata = {
            "class": target_class,
            "class_id": self.CLASS_TO_ID[target_class],
        }

        return result, metadata

    def create_balanced_samples(
        self,
        image: np.ndarray,
    ) -> list[tuple[np.ndarray, dict]]:
        """
        Create one sample for each orientation class from a single image.

        Applies only clean rotations:
        - correct: original image (no rotation)
        - rotate_90_cw: 90° clockwise
        - rotate_180: 180°
        - rotate_90_ccw: 90° counter-clockwise

        Args:
            image: Input correctly-oriented image

        Returns:
            List of (image, metadata) tuples, one per class
        """
        samples = []

        for target_class in self.CLASSES:
            sample, meta = self.augment(image, target_class)
            samples.append((sample, meta))

        return samples

