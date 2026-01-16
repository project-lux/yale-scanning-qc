algo goals:
- Out of focus (YOLO - binary classification)
- Obstruction (e.g hand in image) (YOLO - object detection)
- Image skewed
  - Align to page (default - Algo to handle documents and photographs alike)
  - SOLUTION: Multi-method algorithmic approach:
    1. Hough Line Transform - detects dominant lines
    2. Projection Profile - analyzes text line alignment  
    3. Contour-based - finds document boundaries
    4. Perspective correction via corner detection
- Orientation incorrect
  - 90-degree issue (YOLO - classification for turn 180, turn 90 left, turn 90 right, or perfect orientation)
- Dust in image (YOLO - binary classification)
