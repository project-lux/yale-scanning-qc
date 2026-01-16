"""
Background augmentation - places images on black background with border.
Simulates how documents appear on scanner bed.
"""

import random
from typing import Optional

import cv2
import numpy as np


class BackgroundAugmenter:
    """
    Places images on a black background with configurable border.
    Simulates the appearance of scanned documents on a scanner bed.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        np.random.seed(seed)

    def add_black_background(
        self,
        image: np.ndarray,
        border_size: int = 50,
        border_variation: int = 20,
        randomize_position: bool = True,
    ) -> np.ndarray:
        """
        Place image on black background with border.

        Args:
            image: Input image (BGR)
            border_size: Base border size in pixels
            border_variation: Random variation in border (+/-)
            randomize_position: Whether to slightly randomize image position

        Returns:
            Image on black background
        """
        h, w = image.shape[:2]

        # Calculate borders with optional variation
        if randomize_position:
            top = border_size + self.rng.randint(-border_variation, border_variation)
            bottom = border_size + self.rng.randint(-border_variation, border_variation)
            left = border_size + self.rng.randint(-border_variation, border_variation)
            right = border_size + self.rng.randint(-border_variation, border_variation)
        else:
            top = bottom = left = right = border_size

        # Ensure positive borders
        top = max(10, top)
        bottom = max(10, bottom)
        left = max(10, left)
        right = max(10, right)

        # Create black canvas
        new_h = h + top + bottom
        new_w = w + left + right
        canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)

        # Place image on canvas
        canvas[top:top + h, left:left + w] = image

        return canvas

    def add_scanner_simulation(
        self,
        image: np.ndarray,
        border_size: int = 50,
        border_variation: int = 20,
        add_slight_rotation: bool = True,
        max_rotation: float = 2.0,
        add_shadow: bool = True,
    ) -> np.ndarray:
        """
        Simulate realistic scanner appearance with black background.

        Args:
            image: Input image
            border_size: Base border size
            border_variation: Border size variation
            add_slight_rotation: Add slight random rotation (as if placed slightly off)
            max_rotation: Maximum rotation in degrees
            add_shadow: Add subtle shadow around document edge

        Returns:
            Scanner-simulated image
        """
        h, w = image.shape[:2]

        # Apply slight rotation if enabled
        if add_slight_rotation:
            angle = self.rng.uniform(-max_rotation, max_rotation)
            center = (w // 2, h // 2)
            
            # Calculate new bounding box
            cos_a = abs(np.cos(np.radians(angle)))
            sin_a = abs(np.sin(np.radians(angle)))
            new_w = int(h * sin_a + w * cos_a)
            new_h = int(h * cos_a + w * sin_a)
            
            # Rotation matrix
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            matrix[0, 2] += (new_w - w) / 2
            matrix[1, 2] += (new_h - h) / 2
            
            # Rotate with black background
            rotated = cv2.warpAffine(
                image, matrix, (new_w, new_h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )
        else:
            rotated = image

        # Add shadow around edges if enabled
        if add_shadow:
            rotated = self._add_edge_shadow(rotated)

        # Place on black background with border
        result = self.add_black_background(
            rotated,
            border_size=border_size,
            border_variation=border_variation,
            randomize_position=True,
        )

        return result

    def _add_edge_shadow(self, image: np.ndarray, shadow_size: int = 15) -> np.ndarray:
        """Add subtle shadow around document edges."""
        result = image.copy()
        h, w = image.shape[:2]

        # Create mask of non-black pixels (the document)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # Dilate to get shadow region
        kernel = np.ones((shadow_size, shadow_size), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        shadow_mask = dilated - mask

        # Apply shadow (darken the shadow region slightly)
        shadow_mask_3ch = cv2.cvtColor(shadow_mask, cv2.COLOR_GRAY2BGR) / 255.0
        shadow_color = np.array([20, 20, 20], dtype=np.float32)
        
        # Blur shadow for softness
        shadow_mask_blurred = cv2.GaussianBlur(shadow_mask_3ch, (21, 21), 7)
        
        result = result.astype(np.float32)
        result = result * (1 - shadow_mask_blurred * 0.3)
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def augment(
        self,
        image: np.ndarray,
        border_size: int = 50,
        simulate_scanner: bool = True,
    ) -> tuple[np.ndarray, dict]:
        """
        Apply black background augmentation.

        Args:
            image: Input image
            border_size: Border size in pixels
            simulate_scanner: Whether to add scanner simulation effects

        Returns:
            Tuple of (augmented image, metadata)
        """
        if simulate_scanner:
            result = self.add_scanner_simulation(
                image,
                border_size=border_size,
                border_variation=border_size // 3,
            )
            metadata = {
                "background": "scanner_simulation",
                "border_size": border_size,
            }
        else:
            result = self.add_black_background(
                image,
                border_size=border_size,
            )
            metadata = {
                "background": "black",
                "border_size": border_size,
            }

        return result, metadata

