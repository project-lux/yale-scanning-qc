from PIL import Image
import os

input_dir = "/Users/wjm55/Downloads/BRBL AI Project Test 2/Hands test"
output_dir = "/Users/wjm55/Downloads/BRBL AI Project Test 2/Hands test resize"
os.makedirs(output_dir, exist_ok=True)

for filename in sorted(os.listdir(input_dir)):
    if not filename.lower().endswith((".tif", ".tiff")):
        continue

    input_path = os.path.join(input_dir, filename)
    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f"{base_name}.jpg")

    with Image.open(input_path) as img:
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")

        MAX_SIDE = 2500
        w, h = img.size
        if w > MAX_SIDE or h > MAX_SIDE:
            scale = MAX_SIDE / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), resample=Image.LANCZOS)

        img.save(output_path, "JPEG", quality=85)

    print(f"Saved {output_path}")

print(f"\nDone. Converted {len([f for f in os.listdir(input_dir) if f.lower().endswith(('.tif', '.tiff'))])} images to '{output_dir}'.")
