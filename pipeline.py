"""
Conveyor Belt Damage Detection Pipeline

This script detects damage (scratches and edge_damage) on conveyor belts.
Approach:
1. First crop the belt region using the belt_roi model
2. Apply image processing techniques to detect scratches and edge damage
3. Output annotated images and detection JSON files

Usage:
    python pipeline.py --image_dir <path_to_image_folder> --output_dir <folder>
"""

import os
import cv2
import json
import argparse
import numpy as np
from ultralytics import YOLO


def crop_belt(image, belt_model):
    """Crop belt region from image using belt_roi model."""
    if isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            raise ValueError("Failed to read image from path")

    results = belt_model(image)[0]

    if len(results.boxes) == 0:
        return image, (0, 0)

    box = results.boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    h, w = image.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return image, (0, 0)

    return image[y1:y2, x1:x2], (x1, y1)


def detect_scratches(cropped_belt, offset_x=0, offset_y=0):
    """Detect scratches on belt surface using image processing."""
    gray = cv2.cvtColor(cropped_belt, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    edges = cv2.Canny(enhanced, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    scratch_boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        if area < 100 or area > 20000:
            continue

        aspect_ratio = float(bw) / bh if bh > 0 else 0
        if aspect_ratio > 2 or aspect_ratio < 0.5:
            x_min = x + offset_x
            y_min = y + offset_y
            x_max = x + bw + offset_x
            y_max = y + bh + offset_y
            scratch_boxes.append((x_min, y_min, x_max, y_max, 'scratch'))

    return scratch_boxes


def detect_edge_damage(cropped_belt, offset_x=0, offset_y=0):
    """Detect edge damage on belt edges using image processing."""
    gray = cv2.cvtColor(cropped_belt, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    edge_boxes = []
    margin = 30

    if h > margin * 2:
        top_edge = gray[:margin, :]
        top_edges = cv2.Canny(top_edge, 30, 100)
        if np.sum(top_edges) > 2000:
            edge_boxes.append((offset_x, offset_y, offset_x + w, offset_y + margin + 15, 'edge_damage'))

        bottom_edge = gray[-margin:, :]
        bottom_edges = cv2.Canny(bottom_edge, 30, 100)
        if np.sum(bottom_edges) > 2000:
            edge_boxes.append((offset_x, offset_y + h - margin - 15, offset_x + w, offset_y + h, 'edge_damage'))

    return edge_boxes


def run_pipeline(image_dir, output_dir):
    """Run damage detection pipeline on all images."""
    os.makedirs(output_dir, exist_ok=True)

    belt_model = YOLO("runs/detect/train/weights/best.pt")

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        cropped, (offset_x, offset_y) = crop_belt(image, belt_model)

        scratches = detect_scratches(cropped, offset_x, offset_y)
        edge_damages = detect_edge_damage(cropped, offset_x, offset_y)

        all_detections = scratches + edge_damages

        detections = {}
        count = 1

        for (x_min, y_min, x_max, y_max, damage_type) in all_detections:
            color = (0, 0, 255) if damage_type == 'scratch' else (0, 255, 255)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

            detections[str(count)] = {
                "bbox_coordinates": [x_min, y_min, x_max, y_max],
                "damage_type": damage_type
            }
            count += 1

        out_img_path = os.path.join(output_dir, img_name)
        cv2.imwrite(out_img_path, image)

        json_name = f"{os.path.splitext(img_name)[0]}.json"
        json_path = os.path.join(output_dir, json_name)
        with open(json_path, "w") as f:
            json.dump(detections, f, indent=4)

    print(f"Processed {len(image_files)} images. Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conveyor Belt Damage Detection Pipeline")
    parser.add_argument("--image_dir", required=True, help="Path to input image folder")
    parser.add_argument("--output_dir", required=True, help="Path to output folder")

    args = parser.parse_args()

    run_pipeline(args.image_dir, args.output_dir)