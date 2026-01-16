"""
Flask web application for image QC analysis.
Provides a UI with green/red status lights for each QC check.
Shows detection results with color-coded bounding boxes.
"""

import base64
import io
import os
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request
from PIL import Image

from .config import DETECTION_COLORS
from .qc.analyzer import QCAnalyzer

app = Flask(__name__)
analyzer = None  # Will be loaded on startup


def init_analyzer():
    """Initialize the QC analyzer with all models loaded into memory."""
    global analyzer
    if analyzer is None:
        print("\n" + "=" * 50)
        print("Initializing QC Analyzer...")
        print("=" * 50)
        analyzer = QCAnalyzer(preload=True)
        print("=" * 50)
        print("QC Analyzer ready! All models in memory.")
        print("=" * 50 + "\n")
    return analyzer


def get_analyzer():
    """Get the QC analyzer instance."""
    global analyzer
    if analyzer is None:
        return init_analyzer()
    return analyzer


def draw_detections(image: np.ndarray, results: dict) -> np.ndarray:
    """
    Draw detection bounding boxes on the image with labels.
    
    Args:
        image: Input image (BGR format)
        results: QC analysis results dict
        
    Returns:
        Image with detections drawn
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # Draw obstruction detections
    obstruction_results = results.get('obstruction', {})
    obstruction_detections = obstruction_results.get('detections', [])
    
    for det in obstruction_detections:
        class_name = det.get('class', 'unknown')
        confidence = det.get('confidence', 0)
        bbox = det.get('bbox', [])  # xyxy format
        
        if len(bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            # Get color for this class (BGR format)
            color = DETECTION_COLORS.get(class_name, (0, 255, 0))
            
            # Draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label background
            label = f"{class_name}: {confidence:.0%}"
            (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Position label above box, or inside if at top
            label_y = y1 - 5 if y1 > 30 else y1 + label_h + 5
            
            cv2.rectangle(annotated, 
                         (x1, label_y - label_h - 5), 
                         (x1 + label_w + 6, label_y + 5), 
                         color, -1)
            
            # Draw label text (white on colored background)
            cv2.putText(annotated, label, (x1 + 3, label_y), 
                       font, font_scale, (255, 255, 255), thickness)
    
    # Draw dust detections
    dust_results = results.get('dust', {})
    dust_detections = dust_results.get('detections', [])
    
    for det in dust_detections:
        class_name = det.get('class', 'unknown')
        confidence = det.get('confidence', 0)
        bbox = det.get('bbox', [])  # xyxy format
        
        if len(bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            # Get color for this class (BGR format)
            color = DETECTION_COLORS.get(class_name, (255, 0, 255))
            
            # Draw rectangle with slightly thinner line for dust artifacts
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, max(1, thickness - 1))
            
            # Draw small label
            label = f"{class_name}: {confidence:.0%}"
            (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale * 0.8, 1)
            
            # Position label
            label_y = y1 - 3 if y1 > 20 else y2 + label_h + 3
            
            cv2.rectangle(annotated, 
                         (x1, label_y - label_h - 2), 
                         (x1 + label_w + 4, label_y + 2), 
                         color, -1)
            
            cv2.putText(annotated, label, (x1 + 2, label_y - 1), 
                       font, font_scale * 0.8, (255, 255, 255), 1)
    
    return annotated


# Detection color legend for the UI
def get_color_legend():
    """Return color legend HTML for detection classes."""
    legend_items = []
    
    # Obstruction classes
    legend_items.append(('Obstruction', None, True))  # Header
    for class_name in ['hand', 'finger', 'shadow', 'object']:
        color = DETECTION_COLORS.get(class_name, (0, 255, 0))
        # Convert BGR to RGB hex
        hex_color = f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"
        legend_items.append((class_name, hex_color, False))
    
    # Dust classes
    legend_items.append(('Dust/Debris', None, True))  # Header
    for class_name in ['dust_speck', 'scratch', 'smudge', 'fiber']:
        color = DETECTION_COLORS.get(class_name, (255, 0, 255))
        hex_color = f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"
        legend_items.append((class_name, hex_color, False))
    
    return legend_items


# HTML Template with modern UI - Sidebar Layout + Detection Visualization
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Yale Scanning QC</title>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Space+Grotesk:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-tertiary: #1a1a25;
            --accent-cyan: #00f5d4;
            --accent-magenta: #f72585;
            --accent-yellow: #ffc300;
            --text-primary: #e8e8ed;
            --text-secondary: #8888a0;
            --success: #00f5a0;
            --danger: #ff3366;
            --warning: #ffaa00;
            --unknown: #6677aa;
            --border-subtle: rgba(255,255,255,0.08);
            --sidebar-width: 300px;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Space Grotesk', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
        }

        .app-layout {
            display: flex;
            min-height: 100vh;
        }

        /* Left Sidebar */
        .sidebar {
            width: var(--sidebar-width);
            background: var(--bg-secondary);
            border-right: 1px solid var(--border-subtle);
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
            position: fixed;
            top: 0;
            left: 0;
            height: 100vh;
            overflow-y: auto;
        }

        .sidebar-header {
            text-align: center;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border-subtle);
        }

        .sidebar-header h1 {
            font-size: 1.4rem;
            font-weight: 600;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-magenta));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .sidebar-header .subtitle {
            color: var(--text-secondary);
            font-size: 0.8rem;
            margin-top: 0.25rem;
        }

        .upload-zone {
            background: var(--bg-tertiary);
            border: 2px dashed var(--border-subtle);
            border-radius: 12px;
            padding: 1.5rem 1rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
        }

        .upload-zone:hover {
            border-color: var(--accent-cyan);
        }

        .upload-zone.dragover {
            border-color: var(--accent-magenta);
            background: rgba(247, 37, 133, 0.1);
        }

        .upload-zone input {
            position: absolute;
            inset: 0;
            opacity: 0;
            cursor: pointer;
        }

        .upload-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }

        .upload-text {
            font-size: 0.95rem;
            color: var(--text-primary);
            margin-bottom: 0.25rem;
        }

        .upload-hint {
            color: var(--text-secondary);
            font-size: 0.75rem;
        }

        .settings-section {
            background: var(--bg-tertiary);
            border-radius: 12px;
            padding: 1rem;
        }

        .settings-title {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-secondary);
            margin-bottom: 0.75rem;
        }

        .setting-item {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .setting-item label {
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-primary);
        }

        .threshold-control {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .threshold-control input[type="range"] {
            flex: 1;
            height: 4px;
            background: var(--bg-secondary);
            border-radius: 2px;
            outline: none;
            -webkit-appearance: none;
        }

        .threshold-control input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 14px;
            height: 14px;
            background: var(--accent-cyan);
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 0 8px var(--accent-cyan);
        }

        .threshold-control span {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--accent-cyan);
            min-width: 40px;
        }

        .setting-hint {
            font-size: 0.7rem;
            color: var(--text-secondary);
            line-height: 1.4;
        }

        /* Color Legend */
        .legend-section {
            background: var(--bg-tertiary);
            border-radius: 12px;
            padding: 1rem;
        }

        .legend-header {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
            padding-bottom: 0.25rem;
            border-bottom: 1px solid var(--border-subtle);
        }

        .legend-items {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.75rem;
        }

        .legend-color {
            width: 12px;
            height: 12px;
            border-radius: 3px;
            flex-shrink: 0;
        }

        .legend-label {
            text-transform: capitalize;
            color: var(--text-secondary);
        }

        /* Main Content */
        .main-content {
            flex: 1;
            margin-left: var(--sidebar-width);
            padding: 2rem;
            background-image: 
                radial-gradient(ellipse at 20% 20%, rgba(0, 245, 212, 0.03) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 80%, rgba(247, 37, 133, 0.03) 0%, transparent 50%);
        }

        .loading {
            display: none;
            text-align: center;
            padding: 4rem;
        }

        .loading.active {
            display: block;
        }

        .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid var(--bg-tertiary);
            border-top-color: var(--accent-cyan);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .results-container {
            display: none;
        }

        .results-container.active {
            display: block;
            animation: fadeIn 0.4s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        /* Overall Status - Compact */
        .overall-status {
            display: inline-flex;
            align-items: center;
            gap: 1rem;
            padding: 1rem 1.5rem;
            background: var(--bg-secondary);
            border-radius: 12px;
            border: 1px solid var(--border-subtle);
            margin-bottom: 1.5rem;
        }

        .overall-status.pass {
            border-color: var(--success);
            background: rgba(0, 245, 160, 0.05);
        }

        .overall-status.fail {
            border-color: var(--danger);
            background: rgba(255, 51, 102, 0.05);
        }

        .overall-status.partial {
            border-color: var(--warning);
            background: rgba(255, 170, 0, 0.05);
        }

        .overall-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-secondary);
        }

        .overall-value {
            font-size: 1.5rem;
            font-weight: 600;
        }

        .overall-status.pass .overall-value { color: var(--success); }
        .overall-status.fail .overall-value { color: var(--danger); }
        .overall-status.partial .overall-value { color: var(--warning); }

        /* QC Checks - All 5 side by side */
        .checks-row {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 1rem;
            margin-bottom: 2rem;
        }

        @media (max-width: 1200px) {
            .checks-row {
                grid-template-columns: repeat(3, 1fr);
            }
        }

        @media (max-width: 800px) {
            .checks-row {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        .check-card {
            background: var(--bg-secondary);
            border-radius: 10px;
            padding: 1rem;
            border: 1px solid var(--border-subtle);
            transition: all 0.3s ease;
        }

        .check-card:hover {
            border-color: var(--accent-cyan);
        }

        .check-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.75rem;
        }

        .status-light {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            flex-shrink: 0;
        }

        .status-light.pass {
            background: var(--success);
            box-shadow: 0 0 12px var(--success);
        }

        .status-light.fail {
            background: var(--danger);
            box-shadow: 0 0 12px var(--danger);
        }

        .status-light.unknown {
            background: var(--unknown);
            box-shadow: 0 0 12px var(--unknown);
        }

        .check-name {
            font-size: 0.9rem;
            font-weight: 500;
            text-transform: capitalize;
        }

        .check-details {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
            min-height: 2.5em;
        }

        .confidence-bar {
            height: 3px;
            background: var(--bg-tertiary);
            border-radius: 2px;
            overflow: hidden;
        }

        .confidence-fill {
            height: 100%;
            border-radius: 2px;
            transition: width 0.5s ease;
        }

        .confidence-fill.pass { background: var(--success); }
        .confidence-fill.fail { background: var(--danger); }
        .confidence-fill.unknown { background: var(--unknown); }

        /* Images Section */
        .images-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }

        .images-section.single-image {
            grid-template-columns: 1fr;
            max-width: 900px;
        }

        @media (max-width: 1000px) {
            .images-section {
                grid-template-columns: 1fr;
            }
        }

        .image-card {
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid var(--border-subtle);
        }

        .image-card h3 {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-secondary);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .detection-count {
            background: var(--accent-magenta);
            color: white;
            padding: 0.15rem 0.5rem;
            border-radius: 10px;
            font-size: 0.7rem;
            font-weight: 600;
        }

        .preview-image {
            width: 100%;
            border-radius: 8px;
            max-height: 70vh;
            object-fit: contain;
            background: var(--bg-primary);
        }

        .correction-btn {
            margin-top: 1rem;
            padding: 0.6rem 1.2rem;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-magenta));
            border: none;
            border-radius: 8px;
            color: var(--bg-primary);
            font-family: inherit;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
        }

        .correction-btn:hover {
            transform: scale(1.02);
            box-shadow: 0 4px 15px rgba(0, 245, 212, 0.3);
        }

        .empty-state {
            text-align: center;
            padding: 4rem 2rem;
            color: var(--text-secondary);
        }

        .empty-state .icon {
            font-size: 4rem;
            margin-bottom: 1rem;
            opacity: 0.5;
        }

        .empty-state p {
            font-size: 1.1rem;
        }

        /* Toggle for annotated view */
        .view-toggle {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }

        .view-toggle button {
            padding: 0.4rem 0.8rem;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
            color: var(--text-secondary);
            font-family: inherit;
            font-size: 0.75rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .view-toggle button.active {
            background: var(--accent-cyan);
            color: var(--bg-primary);
            border-color: var(--accent-cyan);
        }

        .view-toggle button:hover:not(.active) {
            border-color: var(--accent-cyan);
            color: var(--text-primary);
        }
    </style>
</head>
<body>
    <div class="app-layout">
        <!-- Left Sidebar -->
        <aside class="sidebar">
            <div class="sidebar-header">
                <h1>Yale Scanning QC</h1>
                <p class="subtitle">Image Quality Control</p>
            </div>

            <div class="upload-zone" id="uploadZone">
                <input type="file" id="fileInput" accept="image/*">
                <div class="upload-icon">📸</div>
                <div class="upload-text">Drop image or click</div>
                <div class="upload-hint">JPG, PNG, WebP</div>
            </div>

            <div class="settings-section">
                <div class="settings-title">Settings</div>
                <div class="setting-item">
                    <label for="skewThreshold">Skew Threshold</label>
                    <div class="threshold-control">
                        <input type="range" id="skewThreshold" min="50" max="100" value="95" step="1">
                        <span id="skewThresholdValue">95%</span>
                    </div>
                    <p class="setting-hint">Confidence below this = fail</p>
                </div>
            </div>

            <!-- Color Legend -->
            <div class="legend-section">
                <div class="settings-title">Detection Legend</div>
                
                <div class="legend-header">Obstruction</div>
                <div class="legend-items">
                    <div class="legend-item">
                        <div class="legend-color" style="background: #ff6400;"></div>
                        <span class="legend-label">hand</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #ffa500;"></div>
                        <span class="legend-label">finger</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #808080;"></div>
                        <span class="legend-label">shadow</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #ff0000;"></div>
                        <span class="legend-label">object</span>
                    </div>
                </div>

                <div class="legend-header" style="margin-top: 0.75rem;">Dust/Debris</div>
                <div class="legend-items">
                    <div class="legend-item">
                        <div class="legend-color" style="background: #ff00ff;"></div>
                        <span class="legend-label">dust speck</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #00ffff;"></div>
                        <span class="legend-label">scratch</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #ffff00;"></div>
                        <span class="legend-label">smudge</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #0000ff;"></div>
                        <span class="legend-label">fiber</span>
                    </div>
                </div>
            </div>

            <button class="correction-btn" id="correctBtn" style="display: none;">
                Apply Corrections
            </button>
        </aside>

        <!-- Main Content -->
        <main class="main-content">
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing image...</p>
            </div>

            <div class="empty-state" id="emptyState">
                <div class="icon">📷</div>
                <p>Upload an image to analyze</p>
            </div>

            <div class="results-container" id="results">
                <!-- Overall Status -->
                <div class="overall-status" id="overallStatus">
                    <div class="overall-label">Status</div>
                    <div class="overall-value" id="overallValue">-</div>
                </div>

                <!-- QC Checks - All 5 side by side -->
                <div class="checks-row" id="checksGrid">
                    <!-- Filled by JavaScript -->
                </div>

                <!-- Images below -->
                <div class="images-section" id="imagesSection">
                    <div class="image-card">
                        <h3>
                            <span>Original Image</span>
                        </h3>
                        <img id="previewImage" class="preview-image" src="" alt="Preview">
                    </div>
                    <div class="image-card" id="annotatedCard">
                        <h3>
                            <span>Detected Issues</span>
                            <span class="detection-count" id="detectionCount">0</span>
                        </h3>
                        <img id="annotatedImage" class="preview-image" src="" alt="Annotated">
                    </div>
                </div>
            </div>
        </main>
    </div>

    <script>
        const uploadZone = document.getElementById('uploadZone');
        const fileInput = document.getElementById('fileInput');
        const loading = document.getElementById('loading');
        const emptyState = document.getElementById('emptyState');
        const results = document.getElementById('results');
        const checksGrid = document.getElementById('checksGrid');
        const overallStatus = document.getElementById('overallStatus');
        const overallValue = document.getElementById('overallValue');
        const previewImage = document.getElementById('previewImage');
        const annotatedImage = document.getElementById('annotatedImage');
        const annotatedCard = document.getElementById('annotatedCard');
        const detectionCount = document.getElementById('detectionCount');
        const imagesSection = document.getElementById('imagesSection');
        const correctBtn = document.getElementById('correctBtn');
        const skewThreshold = document.getElementById('skewThreshold');
        const skewThresholdValue = document.getElementById('skewThresholdValue');

        let currentFile = null;
        let currentResults = null;

        // Threshold slider
        skewThreshold.addEventListener('input', (e) => {
            skewThresholdValue.textContent = e.target.value + '%';
            // Re-display results with new threshold if we have results
            if (currentResults) {
                displayResults(currentResults);
            }
        });

        // Drag and drop
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });

        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('dragover');
        });

        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });

        async function handleFile(file) {
            currentFile = file;
            
            // Show preview
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
            };
            reader.readAsDataURL(file);

            // Hide empty state, show loading
            emptyState.style.display = 'none';
            loading.classList.add('active');
            results.classList.remove('active');
            correctBtn.style.display = 'none';

            // Upload and analyze
            const formData = new FormData();
            formData.append('image', file);
            formData.append('skew_threshold', skewThreshold.value / 100);

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                currentResults = data;
                displayResults(data);
            } catch (error) {
                console.error('Error:', error);
                alert('Error analyzing image');
            } finally {
                loading.classList.remove('active');
            }
        }

        function displayResults(data) {
            results.classList.add('active');

            // Get current threshold
            const threshold = parseInt(skewThreshold.value) / 100;

            // Individual checks
            checksGrid.innerHTML = '';
            const checks = ['focus', 'obstruction', 'orientation', 'dust', 'skew'];
            const icons = {
                focus: '🔍',
                obstruction: '✋',
                orientation: '🔄',
                dust: '✨',
                skew: '📐'
            };

            let hasCorrections = false;
            let statuses = [];
            let totalDetections = 0;

            checks.forEach(check => {
                const result = data[check] || { status: 'unknown', confidence: 0, details: 'N/A' };
                let status = result.status || 'unknown';
                const confidence = (result.confidence || 0) * 100;

                // Apply custom threshold for skew
                if (check === 'skew' && result.confidence !== undefined) {
                    if (result.confidence < threshold) {
                        status = 'fail';
                    } else {
                        status = 'pass';
                    }
                }

                // Count detections
                if (result.detections) {
                    totalDetections += result.detections.length;
                }

                statuses.push(status);
                if (status === 'fail') hasCorrections = true;

                const card = document.createElement('div');
                card.className = 'check-card';
                card.innerHTML = `
                    <div class="check-header">
                        <div class="status-light ${status}"></div>
                        <div class="check-name">${icons[check]} ${check}</div>
                    </div>
                    <div class="check-details">${result.details || 'No details'}</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill ${status}" style="width: ${confidence}%"></div>
                    </div>
                `;
                checksGrid.appendChild(card);
            });

            // Update detection count
            detectionCount.textContent = totalDetections;

            // Show/hide annotated image based on detections
            if (totalDetections > 0 && data.annotated_image) {
                annotatedImage.src = 'data:image/jpeg;base64,' + data.annotated_image;
                annotatedCard.style.display = 'block';
                imagesSection.classList.remove('single-image');
            } else {
                annotatedCard.style.display = 'none';
                imagesSection.classList.add('single-image');
            }

            // Recalculate overall status based on thresholds
            let overall = 'pass';
            if (statuses.includes('fail')) {
                overall = 'fail';
            } else if (statuses.includes('unknown')) {
                overall = 'partial';
            }
            overallStatus.className = 'overall-status ' + overall;
            overallValue.textContent = overall.toUpperCase();

            // Show correction button if needed
            if (hasCorrections) {
                correctBtn.style.display = 'inline-block';
            } else {
                correctBtn.style.display = 'none';
            }
        }

        correctBtn.addEventListener('click', async () => {
            if (!currentFile) return;

            correctBtn.textContent = 'Correcting...';
            correctBtn.disabled = true;

            const formData = new FormData();
            formData.append('image', currentFile);

            try {
                const response = await fetch('/correct', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                if (data.corrected_image) {
                    // Replace original with corrected
                    previewImage.src = 'data:image/jpeg;base64,' + data.corrected_image;
                    
                    // Re-analyze corrected image
                    const blob = await (await fetch('data:image/jpeg;base64,' + data.corrected_image)).blob();
                    const correctedFile = new File([blob], 'corrected.jpg', {type: 'image/jpeg'});
                    currentFile = correctedFile;
                    handleFile(correctedFile);
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Error applying corrections');
            } finally {
                correctBtn.textContent = 'Apply Corrections';
                correctBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    """Serve the main page."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze an uploaded image and return results with annotated image."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    # Get skew threshold from request
    skew_threshold = float(request.form.get('skew_threshold', 0.95))

    try:
        # Read image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Could not decode image'}), 400

        # Analyze
        qc = get_analyzer()
        results = qc.analyze(image, skew_threshold=skew_threshold)

        # Draw detections on image
        annotated = draw_detections(image, results)
        
        # Encode annotated image
        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')
        results['annotated_image'] = annotated_base64

        # Calculate overall
        statuses = [r.get('status', 'unknown') for r in results.values() if isinstance(r, dict)]
        if 'fail' in statuses:
            results['overall'] = 'fail'
        elif 'unknown' in statuses:
            results['overall'] = 'partial'
        else:
            results['overall'] = 'pass'

        return jsonify(results)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/correct', methods=['POST'])
def correct():
    """Apply corrections to an uploaded image."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    try:
        # Read image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Could not decode image'}), 400

        qc = get_analyzer()
        corrected = image.copy()
        corrections_applied = []

        # Analyze first
        results = qc.analyze(image)

        # Apply orientation correction
        orientation = results.get('orientation', {})
        if orientation.get('status') == 'fail' and orientation.get('correction'):
            corrected = qc.correct_orientation(corrected, orientation['correction'])
            corrections_applied.append(f"Rotated {orientation['correction']}°")

        # Apply skew correction
        skew = results.get('skew', {})
        if skew.get('status') == 'fail':
            corrected, skew_meta = qc.correct_skew(corrected)
            if skew_meta.get('corrected'):
                corrections_applied.append(f"Deskewed {skew_meta['angle']:.1f}°")

        # Encode result
        _, buffer = cv2.imencode('.jpg', corrected, [cv2.IMWRITE_JPEG_QUALITY, 95])
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'corrected_image': img_base64,
            'corrections': corrections_applied,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def main():
    """Run the Flask app."""
    import argparse

    parser = argparse.ArgumentParser(description="Run QC web app")
    parser.add_argument('-p', '--port', type=int, default=5000, help='Port (default: 5000)')
    parser.add_argument('-d', '--debug', action='store_true', help='Debug mode')
    parser.add_argument('--host', default='127.0.0.1', help='Host (default: 127.0.0.1)')

    args = parser.parse_args()

    # Pre-load all models into memory before starting server
    init_analyzer()

    print(f"\nStarting QC app at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
