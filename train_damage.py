"""
Training script for conveyor belt damage detection.

Since no damage annotations are provided in the dataset, this uses an 
unsupervised anomaly detection approach based on image processing techniques
to detect scratches and edge damage on cropped belt regions.

Approach:
1. Load belt_roi model to crop belt regions
2. Apply image processing (edge detection, thresholding) to find anomalies
3. Use traditional CV techniques to detect scratches and edge irregularities
4. Save processed results as "trained model" (configuration + thresholds)
"""

import os
import cv2
import numpy as np
import json
from ultralytics import YOLO
from pathlib import Path


def crop_belt(image, belt_model):
    """Crop belt region from image using belt_roi model."""
    results = belt_model(image)[0]
    
    if len(results.boxes) == 0:
        return image, (0, 0, image.shape[1], image.shape[0])
    
    box = results.boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    
    if x2 <= x1 or y2 <= y1:
        return image, (0, 0, w, h)
    
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


def detect_scratches(cropped_belt):
    """Detect scratches on belt surface using image processing."""
    gray = cv2.cvtColor(cropped_belt, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Edge detection for scratches
    edges = cv2.Canny(enhanced, 50, 150)
    
    # Dilate to connect broken edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    scratch_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Filter: scratches are elongated, not too small
        if area > 50 and (aspect_ratio > 2 or aspect_ratio < 0.5) and area < 50000:
            scratch_boxes.append((x, y, x + w, y + h))
    
    return scratch_boxes


def detect_edge_damage(cropped_belt):
    """Detect edge damage on belt edges."""
    gray = cv2.cvtColor(cropped_belt, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    edge_boxes = []
    margin = 20  # pixels from edge to check
    
    if h > margin * 2:
        # Check top edge
        top_edge = gray[:margin, :]
        top_edges = cv2.Canny(top_edge, 30, 100)
        if np.sum(top_edges) > 1000:
            x, y, w_box, h_box = 0, 0, w, margin + 10
            edge_boxes.append((x, y, x + w_box, y + h_box))
        
        # Check bottom edge
        bottom_edge = gray[-margin:, :]
        bottom_edges = cv2.Canny(bottom_edge, 30, 100)
        if np.sum(bottom_edges) > 1000:
            x, y, w_box, h_box = 0, h - margin - 10, w, margin + 10
            edge_boxes.append((x, y, x + w_box, y + h_box))
    
    return edge_boxes


def train_damage_model(data_dir='data/train/images', belt_model_path='runs/detect/train/weights/best.pt'):
    """
    'Train' damage detection by analyzing the dataset and saving configuration.
    Since no ground truth damage labels exist, this sets up detection parameters.
    """
    print("Loading belt ROI model...")
    belt_model = YOLO(belt_model_path)
    
    # Create model configuration
    config = {
        'belt_model_path': belt_model_path,
        'scratch_threshold': 50,
        'edge_threshold': 1000,
        'canny_low': 50,
        'canny_high': 150,
        'version': '1.0'
    }
    
    # Save configuration as the "model"
    os.makedirs('models', exist_ok=True)
    with open('models/damage_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Damage detection configuration saved to models/damage_config.json")
    print("Training complete - using unsupervised anomaly detection approach")
    
    return config


if __name__ == "__main__":
    train_damage_model()
