# Download 5,000 images (seed 1337, max 1500px, 20 threads)
uv run yale-qc download -n 5000

# Augment for all QC categories
uv run yale-qc augment

# Prepare datasets for YOLO training
uv run yale-qc prepare

# Train all YOLO26 models
uv run yale-qc train

# Run the web app with green/red QC status lights
uv run yale-qc serve

# Analyze a single image
uv run yale-qc analyze path/to/image.jpg