"""
Quality Control Analyzer - runs all QC checks on an image.
Supports parallel execution for faster analysis.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from ..augmentation.skew import SkewCorrector
from ..config import MODELS_DIR, QC_CATEGORIES


class QCAnalyzer:
    """
    Runs all quality control checks on an image.

    For YOLO-based checks (focus, obstruction, orientation, dust),
    loads trained models. For skew detection, uses algorithmic approach.

    Features:
    - All models loaded into memory on initialization for fast inference
    - Parallel execution of all 5 checks simultaneously
    - Thread-safe design for concurrent requests
    """

    def __init__(self, models_dir: Path = MODELS_DIR, preload: bool = True):
        self.models_dir = models_dir
        self.models = {}
        self.skew_corrector = SkewCorrector()
        self._models_loaded = False

        # Load available models on init
        if preload:
            self._load_models()

    def _load_models(self):
        """Load all available YOLO models into memory."""
        print("Loading QC models into memory...")
        for category, config in QC_CATEGORIES.items():
            if config.get("type") == "algorithmic":
                continue

            model_path = self.models_dir / config["model"]
            if model_path.exists():
                try:
                    self.models[category] = YOLO(str(model_path))
                    print(f"  ✓ Loaded {category} model")
                except Exception as e:
                    print(f"  ✗ Error loading {category} model: {e}")
            else:
                print(f"  - Model not found for {category}: {model_path}")

        self._models_loaded = True
        print(f"Models loaded: {len(self.models)}/4 YOLO + skew algorithm")

    def ensure_models_loaded(self):
        """Ensure models are loaded (call if preload=False was used)."""
        if not self._models_loaded:
            self._load_models()

    @property
    def loaded_models(self) -> list[str]:
        """Return list of loaded model names."""
        return list(self.models.keys()) + ["skew"]

    def analyze_focus(self, image: np.ndarray) -> dict:
        """
        Check if image is in focus.

        Returns:
            Dict with 'status' (pass/fail), 'confidence', and 'details'
        """
        if "focus" not in self.models:
            return {
                "status": "unknown",
                "confidence": 0.0,
                "details": "Focus model not loaded",
            }

        results = self.models["focus"](image, verbose=False)

        if results and len(results) > 0:
            probs = results[0].probs
            if probs is not None:
                class_idx = probs.top1
                confidence = float(probs.top1conf)
                class_name = results[0].names[class_idx]

                return {
                    "status": "pass" if class_name == "in_focus" else "fail",
                    "confidence": confidence,
                    "details": f"Detected: {class_name}",
                    "class": class_name,
                }

        return {
            "status": "unknown",
            "confidence": 0.0,
            "details": "Could not analyze focus",
        }

    def analyze_obstruction(self, image: np.ndarray) -> dict:
        """
        Detect obstructions (hands, fingers, shadows, objects).

        Returns:
            Dict with 'status', 'confidence', 'detections' list
        """
        if "obstruction" not in self.models:
            return {
                "status": "unknown",
                "confidence": 0.0,
                "details": "Obstruction model not loaded",
                "detections": [],
            }

        results = self.models["obstruction"](image, verbose=False)

        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    detection = {
                        "class": results[0].names[int(boxes.cls[i])],
                        "confidence": float(boxes.conf[i]),
                        "bbox": boxes.xyxy[i].tolist(),
                    }
                    detections.append(detection)

        has_obstruction = len(detections) > 0
        max_conf = max([d["confidence"] for d in detections], default=0.0)

        return {
            "status": "fail" if has_obstruction else "pass",
            "confidence": max_conf if has_obstruction else 1.0,
            "details": f"Found {len(detections)} obstructions" if has_obstruction else "No obstructions",
            "detections": detections,
        }

    def analyze_orientation(self, image: np.ndarray) -> dict:
        """
        Check image orientation.

        Returns:
            Dict with 'status', 'confidence', 'correction' needed
        """
        if "orientation" not in self.models:
            return {
                "status": "unknown",
                "confidence": 0.0,
                "details": "Orientation model not loaded",
                "correction": None,
            }

        results = self.models["orientation"](image, verbose=False)

        if results and len(results) > 0:
            probs = results[0].probs
            if probs is not None:
                class_idx = probs.top1
                confidence = float(probs.top1conf)
                class_name = results[0].names[class_idx]

                correction_map = {
                    "correct": None,
                    "rotate_90_cw": 90,
                    "rotate_180": 180,
                    "rotate_90_ccw": -90,
                }

                return {
                    "status": "pass" if class_name == "correct" else "fail",
                    "confidence": confidence,
                    "details": f"Orientation: {class_name}",
                    "class": class_name,
                    "correction": correction_map.get(class_name),
                }

        return {
            "status": "unknown",
            "confidence": 0.0,
            "details": "Could not analyze orientation",
            "correction": None,
        }

    def analyze_dust(self, image: np.ndarray) -> dict:
        """
        Detect dust/debris artifacts using object detection.

        Returns:
            Dict with 'status', 'confidence', 'detections' list
        """
        if "dust" not in self.models:
            return {
                "status": "unknown",
                "confidence": 0.0,
                "details": "Dust model not loaded",
                "detections": [],
            }

        results = self.models["dust"](image, verbose=False)

        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    detection = {
                        "class": results[0].names[int(boxes.cls[i])],
                        "confidence": float(boxes.conf[i]),
                        "bbox": boxes.xyxy[i].tolist(),
                    }
                    detections.append(detection)

        has_dust = len(detections) > 0
        max_conf = max([d["confidence"] for d in detections], default=0.0)

        return {
            "status": "fail" if has_dust else "pass",
            "confidence": max_conf if has_dust else 1.0,
            "details": f"Found {len(detections)} artifacts" if has_dust else "No dust/debris",
            "detections": detections,
        }

    def analyze_skew(self, image: np.ndarray, threshold: float = 0.95) -> dict:
        """
        Detect skew in image using algorithmic approach.

        Args:
            image: Input image
            threshold: Confidence threshold for pass/fail (default 0.95 = 95%)

        Returns:
            Dict with 'status', 'angle', 'method', 'confidence'
        """
        angle, method, confidence = self.skew_corrector.detect_skew(image)

        # Status based on confidence threshold (can be overridden in UI)
        passes = confidence >= threshold

        # Format details with angle and confidence
        conf_pct = confidence * 100
        if abs(angle) > 0.5:
            details = f"Angle: {angle:.2f}° | Confidence: {conf_pct:.0f}%"
        else:
            details = f"Image aligned | Confidence: {conf_pct:.0f}%"

        return {
            "status": "pass" if passes else "fail",
            "confidence": confidence,
            "details": details,
            "angle": angle,
            "method": method,
        }

    def correct_skew(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Correct skew in image.

        Returns:
            Tuple of (corrected image, metadata)
        """
        return self.skew_corrector.correct_skew(image)

    def correct_orientation(self, image: np.ndarray, rotation: int) -> np.ndarray:
        """Apply rotation correction to image."""
        if rotation == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif rotation == -90 or rotation == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    def analyze(self, image: np.ndarray, parallel: bool = True, skew_threshold: float = 0.95) -> dict:
        """
        Run all QC checks on an image.

        Args:
            image: Input image (BGR format)
            parallel: If True, run all checks simultaneously (faster)
            skew_threshold: Confidence threshold for skew pass/fail

        Returns:
            Dict with results for each check category
        """
        if not parallel:
            # Sequential execution
            return {
                "focus": self.analyze_focus(image),
                "obstruction": self.analyze_obstruction(image),
                "orientation": self.analyze_orientation(image),
                "dust": self.analyze_dust(image),
                "skew": self.analyze_skew(image, skew_threshold),
            }

        # Parallel execution - all 5 checks run simultaneously
        results = {}
        checks = {
            "focus": lambda: self.analyze_focus(image),
            "obstruction": lambda: self.analyze_obstruction(image),
            "orientation": lambda: self.analyze_orientation(image),
            "dust": lambda: self.analyze_dust(image),
            "skew": lambda: self.analyze_skew(image, skew_threshold),
        }

        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all tasks
            futures = {
                executor.submit(func): name
                for name, func in checks.items()
            }

            # Collect results as they complete
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    results[name] = {
                        "status": "unknown",
                        "confidence": 0.0,
                        "details": f"Error: {str(e)}",
                    }

        return results

    def analyze_file(self, image_path: Path) -> dict:
        """
        Analyze an image file.

        Args:
            image_path: Path to image file

        Returns:
            Dict with results and image info
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return {"error": f"Could not load image: {image_path}"}

        results = self.analyze(image)
        results["image"] = {
            "path": str(image_path),
            "size": image.shape[:2],
        }

        # Calculate overall status
        statuses = [r["status"] for r in results.values() if isinstance(r, dict) and "status" in r]
        if "fail" in statuses:
            results["overall"] = "fail"
        elif "unknown" in statuses:
            results["overall"] = "partial"
        else:
            results["overall"] = "pass"

        return results

