"""
Image downloader from Yale IIIF manifests with threading support.
Downloads images with max resolution constraint and deterministic sampling.
"""

import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Optional

import requests
from PIL import Image
from tqdm import tqdm

import cv2
import numpy as np

from .config import (
    BLACK_BACKGROUND,
    BORDER_SIZE,
    BORDER_VARIATION,
    DEFAULT_NUM_IMAGES,
    DOWNLOAD_THREADS,
    MAX_RESOLUTION,
    RANDOM_SEED,
    RAW_IMAGES_DIR,
)

# Default manifest file path
DEFAULT_MANIFEST_FILE = Path("/Users/wjm55/data/manifests.json")


def add_black_background(
    image: np.ndarray,
    border_size: int = BORDER_SIZE,
    border_variation: int = BORDER_VARIATION,
    rng: Optional[random.Random] = None,
) -> np.ndarray:
    """
    Place image on black background with border.
    Simulates scanner bed appearance.
    
    Args:
        image: Input image (RGB numpy array)
        border_size: Base border size in pixels
        border_variation: Random variation in border (+/-)
        rng: Random number generator for reproducibility
    
    Returns:
        Image on black background
    """
    if rng is None:
        rng = random.Random()
    
    h, w = image.shape[:2]
    
    # Calculate borders with variation
    top = border_size + rng.randint(-border_variation, border_variation)
    bottom = border_size + rng.randint(-border_variation, border_variation)
    left = border_size + rng.randint(-border_variation, border_variation)
    right = border_size + rng.randint(-border_variation, border_variation)
    
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


def load_manifest_index(manifest_file: Path) -> list[tuple[str, str]]:
    """
    Load the manifest index file.
    
    Format: JSON array of [lux_url, manifest_url] pairs
    """
    with open(manifest_file) as f:
        data = json.load(f)
    
    # Each entry is [lux_url, manifest_url]
    return [(entry[0], entry[1]) for entry in data]


def fetch_iiif_manifest(manifest_url: str, session: requests.Session) -> Optional[dict]:
    """Fetch a IIIF manifest from URL."""
    try:
        response = session.get(manifest_url, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching manifest {manifest_url}: {e}")
        return None


def extract_images_from_manifest(manifest: dict) -> list[dict]:
    """
    Extract image information from a IIIF manifest.
    Supports IIIF Presentation API 2.x and 3.x
    """
    images = []
    
    # IIIF 3.x format (items array)
    if "items" in manifest:
        for canvas in manifest.get("items", []):
            canvas_id = canvas.get("id", "")
            canvas_label = ""
            
            # Get label
            label = canvas.get("label", {})
            if isinstance(label, dict):
                for lang_labels in label.values():
                    if isinstance(lang_labels, list) and lang_labels:
                        canvas_label = lang_labels[0]
                        break
            
            # Get image from annotation pages
            for anno_page in canvas.get("items", []):
                for annotation in anno_page.get("items", []):
                    body = annotation.get("body", {})
                    
                    # Get image service URL
                    service = body.get("service", [])
                    if isinstance(service, list) and service:
                        service = service[0]
                    
                    service_url = service.get("@id", service.get("id", "")) if service else ""
                    
                    if service_url:
                        images.append({
                            "service_url": service_url,
                            "width": body.get("width", 0),
                            "height": body.get("height", 0),
                            "label": canvas_label,
                            "canvas_id": canvas_id,
                        })
                    elif body.get("id"):
                        # Direct image URL
                        images.append({
                            "direct_url": body.get("id"),
                            "width": body.get("width", 0),
                            "height": body.get("height", 0),
                            "label": canvas_label,
                            "canvas_id": canvas_id,
                        })
    
    # IIIF 2.x format (sequences array)
    elif "sequences" in manifest:
        for sequence in manifest.get("sequences", []):
            for canvas in sequence.get("canvases", []):
                canvas_id = canvas.get("@id", "")
                canvas_label = canvas.get("label", "")
                
                for image_anno in canvas.get("images", []):
                    resource = image_anno.get("resource", {})
                    service = resource.get("service", {})
                    
                    service_url = service.get("@id", "")
                    
                    if service_url:
                        images.append({
                            "service_url": service_url,
                            "width": resource.get("width", 0),
                            "height": resource.get("height", 0),
                            "label": canvas_label,
                            "canvas_id": canvas_id,
                        })
                    elif resource.get("@id"):
                        images.append({
                            "direct_url": resource.get("@id"),
                            "width": resource.get("width", 0),
                            "height": resource.get("height", 0),
                            "label": canvas_label,
                            "canvas_id": canvas_id,
                        })
    
    return images


def build_iiif_image_url(image_info: dict, max_resolution: int) -> Optional[str]:
    """Build IIIF Image API URL with resolution constraint."""
    if "direct_url" in image_info:
        return image_info["direct_url"]
    
    if "service_url" not in image_info:
        return None
    
    service_url = image_info["service_url"].rstrip("/")
    
    # Use IIIF size parameter: !w,h means fit within box
    size = f"!{max_resolution},{max_resolution}"
    
    # Build IIIF URL: {service_url}/{region}/{size}/{rotation}/{quality}.{format}
    return f"{service_url}/full/{size}/0/default.jpg"


def generate_filename(image_info: dict, manifest_id: str) -> str:
    """Generate a unique filename for an image."""
    # Extract image ID from service URL
    service_url = image_info.get("service_url", image_info.get("direct_url", ""))
    image_id = service_url.rstrip("/").split("/")[-1]
    
    # Create short hash for uniqueness
    hash_input = f"{manifest_id}_{image_id}"
    short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    
    return f"{image_id}_{short_hash}.jpg"


def download_single_image(
    image_info: dict,
    manifest_id: str,
    output_dir: Path,
    max_resolution: int,
    session: requests.Session,
    add_background: bool = BLACK_BACKGROUND,
    border_size: int = BORDER_SIZE,
    border_variation: int = BORDER_VARIATION,
    rng: Optional[random.Random] = None,
) -> tuple[bool, str]:
    """Download a single image with optional black background."""
    url = build_iiif_image_url(image_info, max_resolution)
    if not url:
        return False, "No URL available"
    
    filename = generate_filename(image_info, manifest_id)
    output_path = output_dir / filename
    
    if output_path.exists():
        return True, "Already exists"
    
    try:
        response = session.get(url, timeout=60)
        response.raise_for_status()
        
        # Verify it's a valid image
        img = Image.open(BytesIO(response.content))
        
        # Ensure max resolution constraint (IIIF should handle this, but verify)
        max_dim = max(img.width, img.height)
        if max_dim > max_resolution:
            scale = max_resolution / max_dim
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Convert to numpy array for background processing
        img_array = np.array(img)
        
        # Add black background if enabled
        if add_background:
            img_array = add_black_background(
                img_array,
                border_size=border_size,
                border_variation=border_variation,
                rng=rng,
            )
        
        # Save using cv2 (converts RGB to BGR automatically when saving)
        # Actually, cv2.imwrite expects BGR, so convert
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True, "Downloaded"
    
    except Exception as e:
        return False, str(e)


def collect_all_images(
    manifest_file: Path,
    session: requests.Session,
    limit: Optional[int] = None,
    seed: int = RANDOM_SEED,
) -> list[dict]:
    """
    Collect image information from all manifests.
    
    Args:
        manifest_file: Path to manifests.json index file
        session: Requests session
        limit: Optional limit on number of manifests to process
        seed: Random seed for sampling
    
    Returns:
        List of image info dicts with manifest_id added
    """
    print(f"Loading manifest index from {manifest_file}...")
    manifest_pairs = load_manifest_index(manifest_file)
    print(f"Found {len(manifest_pairs)} manifests in index")
    
    # Sample manifests if limit specified
    if limit and limit < len(manifest_pairs):
        random.seed(seed)
        manifest_pairs = random.sample(manifest_pairs, limit)
        print(f"Sampled {limit} manifests")
    
    all_images = []
    
    for lux_url, manifest_url in tqdm(manifest_pairs, desc="Fetching manifests"):
        manifest = fetch_iiif_manifest(manifest_url, session)
        if manifest is None:
            continue
        
        # Extract manifest ID from URL
        manifest_id = manifest_url.rstrip("/").split("/")[-1]
        
        images = extract_images_from_manifest(manifest)
        for img in images:
            img["manifest_id"] = manifest_id
            img["manifest_url"] = manifest_url
        
        all_images.extend(images)
    
    return all_images


def download_images(
    num_images: int = DEFAULT_NUM_IMAGES,
    max_resolution: int = MAX_RESOLUTION,
    num_threads: int = DOWNLOAD_THREADS,
    seed: int = RANDOM_SEED,
    manifest_file: Path = DEFAULT_MANIFEST_FILE,
    output_dir: Path = RAW_IMAGES_DIR,
) -> dict:
    """
    Download images from Yale IIIF manifests.
    
    Args:
        num_images: Number of images to download
        max_resolution: Maximum dimension for downloaded images
        num_threads: Number of download threads
        seed: Random seed for reproducible sampling
        manifest_file: Path to manifests.json index file
        output_dir: Directory to save downloaded images
    
    Returns:
        Dictionary with download statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Yale-Scanning-QC/1.0 (Research Project)"
    })
    
    # Estimate how many manifests we need to fetch
    # Assume average of 5 images per manifest
    manifests_to_fetch = min(num_images * 2, num_images // 5 + 100)
    
    # Collect images from manifests
    print("Collecting images from manifests...")
    all_images = collect_all_images(
        manifest_file, session, limit=manifests_to_fetch, seed=seed
    )
    print(f"Found {len(all_images)} total images")
    
    if not all_images:
        print("No images found!")
        return {"success": 0, "failed": 0, "total": 0}
    
    # Sample images
    random.seed(seed)
    if len(all_images) > num_images:
        selected_images = random.sample(all_images, num_images)
    else:
        selected_images = all_images
        print(f"Only {len(all_images)} images available, downloading all")
    
    print(f"\nDownloading {len(selected_images)} images with {num_threads} threads...")
    
    # Download with thread pool
    success_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {}
        for i, img in enumerate(selected_images):
            # Create per-image RNG for reproducible but varied borders
            img_rng = random.Random(seed + i)
            future = executor.submit(
                download_single_image,
                img,
                img["manifest_id"],
                output_dir,
                max_resolution,
                session,
                add_background=True,  # Always add black background
                border_size=BORDER_SIZE,
                border_variation=BORDER_VARIATION,
                rng=img_rng,
            )
            futures[future] = img
        
        with tqdm(total=len(futures), desc="Downloading") as pbar:
            for future in as_completed(futures):
                success, message = future.result()
                if success:
                    success_count += 1
                else:
                    failed_count += 1
                pbar.update(1)
    
    stats = {
        "success": success_count,
        "failed": failed_count,
        "total": len(selected_images),
    }
    
    print(f"\nDownload complete: {success_count} succeeded, {failed_count} failed")
    return stats


def main():
    """CLI entry point for image downloader."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download images from Yale IIIF manifests")
    parser.add_argument(
        "-n", "--num-images",
        type=int,
        default=DEFAULT_NUM_IMAGES,
        help=f"Number of images to download (default: {DEFAULT_NUM_IMAGES})"
    )
    parser.add_argument(
        "-r", "--max-resolution",
        type=int,
        default=MAX_RESOLUTION,
        help=f"Maximum image resolution (default: {MAX_RESOLUTION})"
    )
    parser.add_argument(
        "-t", "--threads",
        type=int,
        default=DOWNLOAD_THREADS,
        help=f"Number of download threads (default: {DOWNLOAD_THREADS})"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for sampling (default: {RANDOM_SEED})"
    )
    parser.add_argument(
        "-m", "--manifest-file",
        type=Path,
        default=DEFAULT_MANIFEST_FILE,
        help=f"Path to manifests.json index (default: {DEFAULT_MANIFEST_FILE})"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=RAW_IMAGES_DIR,
        help=f"Output directory for images (default: {RAW_IMAGES_DIR})"
    )
    
    args = parser.parse_args()
    
    download_images(
        num_images=args.num_images,
        max_resolution=args.max_resolution,
        num_threads=args.threads,
        seed=args.seed,
        manifest_file=args.manifest_file,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
