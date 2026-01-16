"""
Skew detection, correction, and augmentation.

This module provides:
1. SkewAugmenter: Creates skewed versions of images for training
2. SkewCorrector: Detects and corrects skew in images (algorithmic, not ML-based)

The skew correction algorithm uses multiple techniques:
- Hough Line Transform for document edge detection
- Text line detection for documents with text
- Corner detection for perspective correction
- Contour analysis for finding document boundaries
"""

import random
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage


class SkewAugmenter:
    """
    Creates skewed/perspective-transformed versions of images for training.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        np.random.seed(seed)

    def apply_perspective_transform(
        self,
        image: np.ndarray,
        strength: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply random perspective transformation.

        Args:
            image: Input image
            strength: How extreme the perspective shift (0-0.3 recommended)

        Returns:
            Tuple of (transformed image, transformation matrix)
        """
        h, w = image.shape[:2]

        # Original corners
        src_points = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ])

        # Randomly shift corners
        max_shift = int(min(w, h) * strength)
        dst_points = src_points.copy()
        for i in range(4):
            dst_points[i, 0] += self.rng.randint(-max_shift, max_shift)
            dst_points[i, 1] += self.rng.randint(-max_shift, max_shift)

        # Compute perspective transform
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        result = cv2.warpPerspective(
            image, matrix, (w, h),
            borderMode=cv2.BORDER_REPLICATE
        )

        return result, matrix

    def apply_rotation_skew(
        self,
        image: np.ndarray,
        max_angle: float = 15.0,
    ) -> tuple[np.ndarray, float]:
        """
        Apply rotation-based skew.

        Args:
            image: Input image
            max_angle: Maximum rotation angle in degrees

        Returns:
            Tuple of (rotated image, applied angle)
        """
        h, w = image.shape[:2]
        angle = self.rng.uniform(-max_angle, max_angle)

        # Avoid very small angles
        if abs(angle) < 1.0:
            angle = self.rng.choice([-1, 1]) * self.rng.uniform(2, max_angle)

        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        result = cv2.warpAffine(
            image, matrix, (w, h),
            borderMode=cv2.BORDER_REPLICATE
        )

        return result, angle

    def apply_affine_skew(
        self,
        image: np.ndarray,
        shear_range: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply affine (shear) transformation.

        Args:
            image: Input image
            shear_range: Maximum shear factor

        Returns:
            Tuple of (sheared image, transformation matrix)
        """
        h, w = image.shape[:2]

        # Random shear
        shear_x = self.rng.uniform(-shear_range, shear_range)
        shear_y = self.rng.uniform(-shear_range, shear_range)

        # Affine transformation matrix
        src_points = np.float32([[0, 0], [w, 0], [0, h]])
        dst_points = np.float32([
            [w * shear_y, h * shear_x],
            [w * (1 + shear_y), h * shear_x],
            [w * shear_y, h * (1 + shear_x)]
        ])

        matrix = cv2.getAffineTransform(src_points, dst_points)
        result = cv2.warpAffine(
            image, matrix, (w, h),
            borderMode=cv2.BORDER_REPLICATE
        )

        return result, matrix

    def augment(
        self,
        image: np.ndarray,
        skew_type: Optional[str] = None,
        severity: str = "medium",
    ) -> tuple[np.ndarray, dict]:
        """
        Apply skew augmentation.

        Args:
            image: Input image
            skew_type: Type of skew (perspective, rotation, affine, or None for random)
            severity: Skew amount (light, medium, heavy)

        Returns:
            Tuple of (skewed image, metadata dict)
        """
        severity_config = {
            "light": {"perspective": 0.05, "rotation": 5, "affine": 0.05},
            "medium": {"perspective": 0.1, "rotation": 10, "affine": 0.1},
            "heavy": {"perspective": 0.2, "rotation": 20, "affine": 0.15},
        }
        config = severity_config.get(severity, severity_config["medium"])

        if skew_type is None:
            skew_type = self.rng.choice(["perspective", "rotation", "affine"])

        if skew_type == "perspective":
            result, matrix = self.apply_perspective_transform(
                image, strength=config["perspective"]
            )
            metadata = {
                "skew_type": "perspective",
                "strength": config["perspective"],
            }
        elif skew_type == "rotation":
            result, angle = self.apply_rotation_skew(
                image, max_angle=config["rotation"]
            )
            metadata = {
                "skew_type": "rotation",
                "angle": angle,
            }
        elif skew_type == "affine":
            result, matrix = self.apply_affine_skew(
                image, shear_range=config["affine"]
            )
            metadata = {
                "skew_type": "affine",
                "shear_range": config["affine"],
            }
        else:
            result = image.copy()
            metadata = {"skew_type": "none"}

        metadata["class"] = "skewed"
        return result, metadata

    def create_aligned_sample(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        """Return original aligned image for comparison."""
        return image.copy(), {"class": "aligned", "skew_type": "none"}


class SkewCorrector:
    """
    Algorithmic skew detection and correction.

    Uses multiple detection methods and selects the best:
    1. Hough Line Transform - detects dominant lines
    2. Projection Profile - analyzes horizontal projection of text
    3. Contour-based - finds document boundaries
    4. FFT-based - frequency domain analysis

    The corrector automatically detects whether the image is a
    document (text-heavy) or photograph and applies appropriate methods.
    """

    def __init__(self):
        self.methods = ["hough", "projection", "contour"]

    def detect_skew_hough(self, image: np.ndarray) -> Optional[float]:
        """
        Detect skew angle using Hough Line Transform.

        Best for: Documents with clear edges or lines.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Dilate edges to connect broken lines
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Hough Line Transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is None or len(lines) == 0:
            return None

        # Collect angles
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90

            # Normalize to -45 to 45 range
            while angle > 45:
                angle -= 90
            while angle < -45:
                angle += 90

            angles.append(angle)

        if not angles:
            return None

        # Use median to be robust to outliers
        return np.median(angles)

    def detect_skew_projection(self, image: np.ndarray) -> Optional[float]:
        """
        Detect skew using horizontal projection profile.

        Best for: Text documents where horizontal lines of text are expected.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Try different angles and find the one with strongest horizontal projection
        best_angle = 0
        best_score = 0

        for angle in np.linspace(-15, 15, 61):
            rotated = ndimage.rotate(binary, angle, reshape=False, order=0)

            # Horizontal projection (sum of white pixels per row)
            projection = np.sum(rotated, axis=1)

            # Score: variance of projection (higher = more aligned text lines)
            score = np.var(projection)

            if score > best_score:
                best_score = score
                best_angle = angle

        return best_angle if best_score > 0 else None

    def detect_skew_contour(self, image: np.ndarray) -> Optional[float]:
        """
        Detect skew by finding the largest rectangular contour.

        Best for: Scanned documents with clear page boundaries.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Blur and threshold
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find largest contour
        largest = max(contours, key=cv2.contourArea)

        # Get minimum area rectangle
        rect = cv2.minAreaRect(largest)
        angle = rect[2]

        # Normalize angle
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90

        return angle

    def detect_skew(
        self,
        image: np.ndarray,
        method: str = "auto",
    ) -> tuple[float, str, float]:
        """
        Detect skew angle in image.

        Args:
            image: Input image
            method: Detection method (auto, hough, projection, contour)

        Returns:
            Tuple of (angle, method_used, confidence)
        """
        if method == "auto":
            # Try all methods and pick best
            results = {}

            for m in self.methods:
                try:
                    if m == "hough":
                        angle = self.detect_skew_hough(image)
                    elif m == "projection":
                        angle = self.detect_skew_projection(image)
                    elif m == "contour":
                        angle = self.detect_skew_contour(image)
                    else:
                        continue

                    if angle is not None:
                        results[m] = angle
                except Exception:
                    continue

            if not results:
                return 0.0, "none", 0.0

            # Use median of all detected angles
            angles = list(results.values())
            median_angle = np.median(angles)

            # Confidence based on agreement
            if len(angles) > 1:
                std = np.std(angles)
                confidence = max(0, 1 - std / 10)  # Lower std = higher confidence
            else:
                confidence = 0.5

            # Find which method was closest to median
            best_method = min(results.keys(), key=lambda m: abs(results[m] - median_angle))

            return median_angle, best_method, confidence

        else:
            # Use specified method
            if method == "hough":
                angle = self.detect_skew_hough(image)
            elif method == "projection":
                angle = self.detect_skew_projection(image)
            elif method == "contour":
                angle = self.detect_skew_contour(image)
            else:
                angle = None

            if angle is None:
                return 0.0, method, 0.0

            return angle, method, 0.7

    def correct_skew(
        self,
        image: np.ndarray,
        angle: Optional[float] = None,
        method: str = "auto",
    ) -> tuple[np.ndarray, dict]:
        """
        Detect and correct skew in image.

        Args:
            image: Input image
            angle: Specific angle to correct (or None for auto-detection)
            method: Detection method if angle not provided

        Returns:
            Tuple of (corrected image, metadata dict)
        """
        h, w = image.shape[:2]

        if angle is None:
            angle, method_used, confidence = self.detect_skew(image, method)
        else:
            method_used = "manual"
            confidence = 1.0

        # Skip if angle is negligible
        if abs(angle) < 0.5:
            return image.copy(), {
                "corrected": False,
                "angle": angle,
                "method": method_used,
                "confidence": confidence,
                "reason": "angle_negligible",
            }

        # Rotate to correct
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new bounding box size to avoid cropping
        cos_a = abs(matrix[0, 0])
        sin_a = abs(matrix[0, 1])
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)

        # Adjust transformation matrix for new center
        matrix[0, 2] += (new_w - w) / 2
        matrix[1, 2] += (new_h - h) / 2

        corrected = cv2.warpAffine(
            image, matrix, (new_w, new_h),
            borderMode=cv2.BORDER_REPLICATE
        )

        # Optionally crop to remove border artifacts
        # Find the largest inscribed rectangle
        corrected = self._crop_rotated(corrected, image.shape[:2], angle)

        return corrected, {
            "corrected": True,
            "angle": angle,
            "method": method_used,
            "confidence": confidence,
        }

    def _crop_rotated(
        self,
        image: np.ndarray,
        original_size: tuple[int, int],
        angle: float,
    ) -> np.ndarray:
        """Crop rotated image to remove black borders."""
        h, w = image.shape[:2]
        orig_h, orig_w = original_size

        # Calculate the largest rectangle that fits
        angle_rad = abs(np.radians(angle))
        if angle_rad == 0:
            return image

        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Width and height of inscribed rectangle
        if orig_w * sin_a + orig_h * cos_a != 0:
            wr = orig_w * cos_a - orig_h * sin_a
            hr = orig_h * cos_a - orig_w * sin_a

            if wr > 0 and hr > 0:
                # Calculate crop region
                crop_w = min(int(wr), w)
                crop_h = min(int(hr), h)

                x = (w - crop_w) // 2
                y = (h - crop_h) // 2

                return image[y:y+crop_h, x:x+crop_w]

        return image

    def detect_document_corners(
        self,
        image: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Detect the four corners of a document in the image.

        Returns:
            Array of 4 corner points [[x,y], ...] or None if not found
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find largest contour
        largest = max(contours, key=cv2.contourArea)

        # Approximate to polygon
        epsilon = 0.02 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)

        # Check if it's a quadrilateral
        if len(approx) == 4:
            return approx.reshape(4, 2)

        # If not exactly 4 corners, use convex hull and find 4 extreme points
        hull = cv2.convexHull(largest)
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)

        if len(approx) >= 4:
            # Sort by angle from center and pick 4 most corner-like
            center = approx.mean(axis=0)
            angles = np.arctan2(approx[:, 0, 1] - center[0, 1], approx[:, 0, 0] - center[0, 0])
            sorted_idx = np.argsort(angles)

            # Pick 4 evenly spaced points
            n = len(sorted_idx)
            idx = [sorted_idx[i * n // 4] for i in range(4)]
            return approx[idx].reshape(4, 2)

        return None

    def correct_perspective(
        self,
        image: np.ndarray,
        corners: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Apply perspective correction to straighten a document.

        Args:
            image: Input image
            corners: Four corners of document, or None for auto-detection

        Returns:
            Tuple of (corrected image, metadata dict)
        """
        if corners is None:
            corners = self.detect_document_corners(image)

        if corners is None:
            return image.copy(), {
                "corrected": False,
                "reason": "no_corners_detected",
            }

        # Order corners: top-left, top-right, bottom-right, bottom-left
        corners = self._order_corners(corners)

        # Calculate target dimensions
        width_top = np.linalg.norm(corners[0] - corners[1])
        width_bottom = np.linalg.norm(corners[3] - corners[2])
        height_left = np.linalg.norm(corners[0] - corners[3])
        height_right = np.linalg.norm(corners[1] - corners[2])

        max_width = int(max(width_top, width_bottom))
        max_height = int(max(height_left, height_right))

        # Target corners
        dst = np.float32([
            [0, 0],
            [max_width, 0],
            [max_width, max_height],
            [0, max_height]
        ])

        # Compute and apply perspective transform
        matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
        corrected = cv2.warpPerspective(image, matrix, (max_width, max_height))

        return corrected, {
            "corrected": True,
            "corners": corners.tolist(),
            "output_size": [max_width, max_height],
        }

    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners as: top-left, top-right, bottom-right, bottom-left."""
        # Sort by y coordinate
        sorted_by_y = corners[np.argsort(corners[:, 1])]

        # Top two points
        top = sorted_by_y[:2]
        top = top[np.argsort(top[:, 0])]

        # Bottom two points
        bottom = sorted_by_y[2:]
        bottom = bottom[np.argsort(bottom[:, 0])]

        return np.array([top[0], top[1], bottom[1], bottom[0]])

