"""
Blur augmentation for out-of-focus detection training.
Creates various types of blur to simulate focus issues.
"""

import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


class BlurAugmenter:
    """
    Augmenter that adds various types of blur to simulate out-of-focus images.

    Blur types:
    - Gaussian blur: General out-of-focus
    - Motion blur: Camera shake
    - Defocus blur: Lens defocus simulation
    - Partial blur: Only parts of image blurred
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        np.random.seed(seed)

    def gaussian_blur(
        self,
        image: np.ndarray,
        kernel_size: Optional[int] = None,
        sigma: Optional[float] = None,
    ) -> np.ndarray:
        """Apply Gaussian blur."""
        if kernel_size is None:
            kernel_size = self.rng.choice([5, 7, 9, 11, 15, 21, 31])
        if kernel_size % 2 == 0:
            kernel_size += 1

        if sigma is None:
            sigma = self.rng.uniform(1.0, 10.0)

        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    def motion_blur(
        self,
        image: np.ndarray,
        kernel_size: Optional[int] = None,
        angle: Optional[float] = None,
    ) -> np.ndarray:
        """Apply motion blur to simulate camera shake."""
        if kernel_size is None:
            kernel_size = self.rng.choice([9, 15, 21, 31, 45])
        if angle is None:
            angle = self.rng.uniform(0, 360)

        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = np.ones(kernel_size) / kernel_size

        # Rotate kernel
        center = (kernel_size / 2, kernel_size / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(
            kernel,
            rotation_matrix,
            (kernel_size, kernel_size),
            flags=cv2.INTER_LINEAR,
        )

        return cv2.filter2D(image, -1, kernel)

    def defocus_blur(
        self,
        image: np.ndarray,
        radius: Optional[int] = None,
    ) -> np.ndarray:
        """Apply defocus (circular) blur to simulate lens out-of-focus."""
        if radius is None:
            radius = self.rng.randint(3, 15)

        # Create circular kernel
        kernel_size = 2 * radius + 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        cv2.circle(kernel, (radius, radius), radius, 1.0, -1)
        kernel /= kernel.sum()

        return cv2.filter2D(image, -1, kernel)

    def partial_blur(
        self,
        image: np.ndarray,
        blur_fraction: Optional[float] = None,
    ) -> np.ndarray:
        """Apply blur to only part of the image (simulating partial focus)."""
        if blur_fraction is None:
            blur_fraction = self.rng.uniform(0.3, 0.7)

        h, w = image.shape[:2]

        # Create gradient mask
        mask = np.zeros((h, w), dtype=np.float32)

        # Random blur region
        pattern = self.rng.choice(["horizontal", "vertical", "radial", "corner"])

        if pattern == "horizontal":
            start = int(w * self.rng.uniform(0, 0.3))
            end = int(w * self.rng.uniform(0.7, 1.0))
            mask[:, start:end] = 1.0
            # Gradient edges
            for i in range(min(50, start)):
                mask[:, start - i - 1] = (50 - i) / 50
            for i in range(min(50, w - end)):
                mask[:, end + i] = (50 - i) / 50

        elif pattern == "vertical":
            start = int(h * self.rng.uniform(0, 0.3))
            end = int(h * self.rng.uniform(0.7, 1.0))
            mask[start:end, :] = 1.0
            for i in range(min(50, start)):
                mask[start - i - 1, :] = (50 - i) / 50
            for i in range(min(50, h - end)):
                mask[end + i, :] = (50 - i) / 50

        elif pattern == "radial":
            # Center-focused, edges blurred or vice versa
            center_y, center_x = h // 2, w // 2
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            if self.rng.random() > 0.5:
                mask = 1 - (dist / max_dist)  # Center sharp
            else:
                mask = dist / max_dist  # Center blurred

        elif pattern == "corner":
            # One corner blurred
            corner = self.rng.choice(["tl", "tr", "bl", "br"])
            Y, X = np.ogrid[:h, :w]
            if corner == "tl":
                dist = np.sqrt(X**2 + Y**2)
            elif corner == "tr":
                dist = np.sqrt((w - X) ** 2 + Y**2)
            elif corner == "bl":
                dist = np.sqrt(X**2 + (h - Y) ** 2)
            else:
                dist = np.sqrt((w - X) ** 2 + (h - Y) ** 2)
            max_dist = np.sqrt(w**2 + h**2)
            mask = dist / max_dist

        mask = np.clip(mask, 0, 1)

        # Apply blur
        blurred = self.gaussian_blur(image, kernel_size=21, sigma=8.0)

        # Blend
        mask = mask[:, :, np.newaxis]
        result = (image * (1 - mask) + blurred * mask).astype(np.uint8)

        return result

    def augment(
        self,
        image: np.ndarray,
        blur_type: Optional[str] = None,
        severity: str = "medium",
    ) -> tuple[np.ndarray, dict]:
        """
        Apply blur augmentation.

        Args:
            image: Input image (BGR or RGB)
            blur_type: Type of blur (gaussian, motion, defocus, partial, or None for random)
            severity: Blur severity (light, medium, heavy)

        Returns:
            Tuple of (blurred image, metadata dict)
        """
        if blur_type is None:
            blur_type = self.rng.choice(["gaussian", "motion", "defocus", "partial"])

        severity_params = {
            "light": {"kernel_range": (5, 11), "sigma_range": (1.0, 3.0)},
            "medium": {"kernel_range": (11, 21), "sigma_range": (3.0, 7.0)},
            "heavy": {"kernel_range": (21, 45), "sigma_range": (7.0, 15.0)},
        }
        params = severity_params.get(severity, severity_params["medium"])

        if blur_type == "gaussian":
            kernel_size = self.rng.randint(*params["kernel_range"])
            if kernel_size % 2 == 0:
                kernel_size += 1
            sigma = self.rng.uniform(*params["sigma_range"])
            result = self.gaussian_blur(image, kernel_size, sigma)
            metadata = {"blur_type": "gaussian", "kernel_size": kernel_size, "sigma": sigma}

        elif blur_type == "motion":
            kernel_size = self.rng.randint(*params["kernel_range"])
            angle = self.rng.uniform(0, 360)
            result = self.motion_blur(image, kernel_size, angle)
            metadata = {"blur_type": "motion", "kernel_size": kernel_size, "angle": angle}

        elif blur_type == "defocus":
            radius = self.rng.randint(3, 15)
            result = self.defocus_blur(image, radius)
            metadata = {"blur_type": "defocus", "radius": radius}

        elif blur_type == "partial":
            result = self.partial_blur(image)
            metadata = {"blur_type": "partial"}

        else:
            result = image.copy()
            metadata = {"blur_type": "none"}

        metadata["class"] = "out_of_focus"
        return result, metadata

    def create_sharp_sample(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        """Return original sharp image with metadata for 'in_focus' class."""
        # Just return the original image without any modification
        return image.copy(), {"class": "in_focus", "blur_type": "none"}

