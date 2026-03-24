"""
Obstruction augmentation for hand/object detection training.
Overlays realistic obstructions onto images with bounding box annotations.
"""

import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


class ObstructionAugmenter:
    """
    Augmenter that adds synthetic obstructions to images.

    Obstruction types:
    - Hand silhouettes
    - Finger intrusions
    - Shadow overlays
    - Generic object obstructions
    """

    # Hand/finger shapes as simplified polygons (normalized 0-1 coordinates)
    HAND_TEMPLATES = [
        # Open hand from side
        np.array([
            [0.0, 0.3], [0.1, 0.0], [0.15, 0.05], [0.2, 0.0], [0.25, 0.05],
            [0.3, 0.0], [0.35, 0.1], [0.5, 0.15], [0.6, 0.2], [0.7, 0.3],
            [0.8, 0.5], [0.9, 0.7], [1.0, 0.9], [0.8, 1.0], [0.5, 0.95],
            [0.2, 0.9], [0.0, 0.8]
        ], dtype=np.float32),
        # Pointing finger
        np.array([
            [0.0, 0.4], [0.1, 0.3], [0.3, 0.2], [0.5, 0.15], [0.7, 0.1],
            [0.9, 0.05], [1.0, 0.1], [1.0, 0.3], [0.9, 0.35], [0.7, 0.4],
            [0.5, 0.5], [0.3, 0.6], [0.15, 0.7], [0.1, 0.8], [0.0, 0.7]
        ], dtype=np.float32),
        # Thumb from corner
        np.array([
            [0.0, 0.0], [0.3, 0.0], [0.5, 0.1], [0.6, 0.3], [0.5, 0.5],
            [0.4, 0.7], [0.3, 0.8], [0.2, 0.85], [0.1, 0.8], [0.0, 0.6]
        ], dtype=np.float32),
    ]

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        np.random.seed(seed)

    def _generate_skin_color(self) -> tuple[int, int, int]:
        """Generate a random skin tone color (BGR)."""
        skin_tones = [
            (180, 200, 230),  # Light
            (140, 170, 200),  # Medium light
            (100, 140, 180),  # Medium
            (70, 100, 140),   # Medium dark
            (40, 60, 90),     # Dark
        ]
        base = self.rng.choice(skin_tones)
        # Add variation
        variation = tuple(
            max(0, min(255, c + self.rng.randint(-20, 20)))
            for c in base
        )
        return variation

    def _create_hand_mask(
        self,
        shape: tuple[int, int],
        template_idx: Optional[int] = None,
        position: str = "random",
    ) -> tuple[np.ndarray, list[int]]:
        """
        Create a hand-shaped mask and return bounding box.

        Returns:
            Tuple of (mask, [x, y, w, h] bounding box)
        """
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if template_idx is None:
            template_idx = self.rng.randint(0, len(self.HAND_TEMPLATES) - 1)

        template = self.HAND_TEMPLATES[template_idx].copy()

        # Scale and position
        hand_scale = self.rng.uniform(0.15, 0.4)
        hand_w = int(w * hand_scale)
        hand_h = int(h * hand_scale * self.rng.uniform(0.8, 1.2))

        # Position based on entry point
        positions = {
            "left": (0, self.rng.randint(0, h - hand_h)),
            "right": (w - hand_w, self.rng.randint(0, h - hand_h)),
            "top": (self.rng.randint(0, w - hand_w), 0),
            "bottom": (self.rng.randint(0, w - hand_w), h - hand_h),
            "corner_tl": (0, 0),
            "corner_tr": (w - hand_w, 0),
            "corner_bl": (0, h - hand_h),
            "corner_br": (w - hand_w, h - hand_h),
        }

        if position == "random":
            position = self.rng.choice(list(positions.keys()))

        offset_x, offset_y = positions.get(position, positions["corner_br"])

        # Transform template to pixel coordinates
        points = template.copy()
        points[:, 0] = points[:, 0] * hand_w + offset_x
        points[:, 1] = points[:, 1] * hand_h + offset_y

        # Random rotation
        angle = self.rng.uniform(-30, 30)
        center = points.mean(axis=0)
        cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        points = (points - center) @ rotation.T + center

        # Draw filled polygon
        points = points.astype(np.int32)
        cv2.fillPoly(mask, [points], 255)

        # Calculate bounding box
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)

        # Clip to image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]

        return mask, bbox

    def add_hand_obstruction(
        self,
        image: np.ndarray,
        position: str = "random",
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Add a hand obstruction to the image.

        Returns:
            Tuple of (augmented image, list of annotation dicts)
        """
        result = image.copy()
        h, w = image.shape[:2]

        mask, bbox = self._create_hand_mask((h, w), position=position)

        # Generate skin color
        color = self._generate_skin_color()

        # Create colored hand layer
        hand_layer = np.zeros_like(image)
        hand_layer[:] = color

        # Add texture/noise
        noise = np.random.randint(-15, 15, hand_layer.shape, dtype=np.int16)
        hand_layer = np.clip(hand_layer.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Slight blur for realism
        hand_layer = cv2.GaussianBlur(hand_layer, (5, 5), 2)

        # Apply mask
        mask_3ch = mask[:, :, np.newaxis] / 255.0

        # Soften mask edges
        mask_blur = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 5)
        mask_3ch = (mask_blur / 255.0)[:, :, np.newaxis]

        result = (result * (1 - mask_3ch) + hand_layer * mask_3ch).astype(np.uint8)

        annotation = {
            "class": "hand",
            "class_id": 0,
            "bbox": bbox,  # [x, y, w, h]
            "bbox_normalized": [
                bbox[0] / w,
                bbox[1] / h,
                bbox[2] / w,
                bbox[3] / h,
            ],
        }

        return result, [annotation]

    def add_finger_obstruction(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, list[dict]]:
        """Add a finger pointing into the image."""
        result = image.copy()
        h, w = image.shape[:2]

        mask, bbox = self._create_hand_mask((h, w), template_idx=1, position="random")

        color = self._generate_skin_color()
        finger_layer = np.zeros_like(image)
        finger_layer[:] = color

        # Add subtle gradient
        noise = np.random.randint(-10, 10, finger_layer.shape, dtype=np.int16)
        finger_layer = np.clip(finger_layer.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        mask_blur = cv2.GaussianBlur(mask.astype(np.float32), (11, 11), 3)
        mask_3ch = (mask_blur / 255.0)[:, :, np.newaxis]

        result = (result * (1 - mask_3ch) + finger_layer * mask_3ch).astype(np.uint8)

        annotation = {
            "class": "finger",
            "class_id": 1,
            "bbox": bbox,
            "bbox_normalized": [
                bbox[0] / w,
                bbox[1] / h,
                bbox[2] / w,
                bbox[3] / h,
            ],
        }

        return result, [annotation]

    def add_shadow_obstruction(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, list[dict]]:
        """Add a shadow (like from a hand or object) over the image."""
        result = image.copy()
        h, w = image.shape[:2]

        mask, bbox = self._create_hand_mask((h, w), position="random")

        # Expand shadow slightly
        kernel = np.ones((15, 15), np.uint8)
        shadow_mask = cv2.dilate(mask, kernel, iterations=2)

        # Heavy blur for shadow
        shadow_mask = cv2.GaussianBlur(shadow_mask.astype(np.float32), (51, 51), 20)
        shadow_mask = (shadow_mask / 255.0)[:, :, np.newaxis]

        # Darken the shadowed area
        shadow_intensity = self.rng.uniform(0.3, 0.6)
        result = (result * (1 - shadow_mask * shadow_intensity)).astype(np.uint8)

        # Recalculate bbox for shadow
        shadow_binary = (shadow_mask[:, :, 0] > 0.1).astype(np.uint8)
        coords = np.where(shadow_binary > 0)
        if len(coords[0]) > 0:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]

        annotation = {
            "class": "shadow",
            "class_id": 2,
            "bbox": bbox,
            "bbox_normalized": [
                bbox[0] / w,
                bbox[1] / h,
                bbox[2] / w,
                bbox[3] / h,
            ],
        }

        return result, [annotation]

    def add_object_obstruction(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, list[dict]]:
        """Add a generic rectangular object obstruction."""
        result = image.copy()
        h, w = image.shape[:2]

        # Random rectangle
        obj_w = self.rng.randint(int(w * 0.1), int(w * 0.3))
        obj_h = self.rng.randint(int(h * 0.1), int(h * 0.3))

        # Position from edge
        edge = self.rng.choice(["left", "right", "top", "bottom"])
        if edge == "left":
            x = -obj_w // 2
            y = self.rng.randint(0, h - obj_h)
        elif edge == "right":
            x = w - obj_w // 2
            y = self.rng.randint(0, h - obj_h)
        elif edge == "top":
            x = self.rng.randint(0, w - obj_w)
            y = -obj_h // 2
        else:
            x = self.rng.randint(0, w - obj_w)
            y = h - obj_h // 2

        # Create object mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (x, y), (x + obj_w, y + obj_h), 255, -1)

        # Random dark color for object
        color = tuple(self.rng.randint(20, 80) for _ in range(3))
        obj_layer = np.zeros_like(image)
        obj_layer[:] = color

        mask_blur = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 2)
        mask_3ch = (mask_blur / 255.0)[:, :, np.newaxis]

        result = (result * (1 - mask_3ch) + obj_layer * mask_3ch).astype(np.uint8)

        # Calculate visible bbox
        x_vis = max(0, x)
        y_vis = max(0, y)
        x2_vis = min(w, x + obj_w)
        y2_vis = min(h, y + obj_h)
        bbox = [x_vis, y_vis, x2_vis - x_vis, y2_vis - y_vis]

        annotation = {
            "class": "object",
            "class_id": 3,
            "bbox": bbox,
            "bbox_normalized": [
                bbox[0] / w,
                bbox[1] / h,
                bbox[2] / w,
                bbox[3] / h,
            ],
        }

        return result, [annotation]

    def augment(
        self,
        image: np.ndarray,
        obstruction_type: Optional[str] = None,
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Apply obstruction augmentation.

        Args:
            image: Input image
            obstruction_type: Type of obstruction (hand, finger, shadow, object, or None for random)

        Returns:
            Tuple of (augmented image, list of annotation dicts with bboxes)
        """
        if obstruction_type is None:
            obstruction_type = self.rng.choice(["hand", "finger", "shadow", "object"])

        if obstruction_type == "hand":
            return self.add_hand_obstruction(image)
        elif obstruction_type == "finger":
            return self.add_finger_obstruction(image)
        elif obstruction_type == "shadow":
            return self.add_shadow_obstruction(image)
        elif obstruction_type == "object":
            return self.add_object_obstruction(image)
        else:
            return image, []

    def create_clean_sample(self, image: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """Return original image with no obstructions (negative sample)."""
        return image.copy(), []



