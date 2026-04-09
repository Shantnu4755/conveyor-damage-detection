# 🔧 Conveyor Belt Damage Detection System

An intelligent computer vision system that automatically detects **scratches** and **edge damage** on conveyor belts using advanced image processing techniques.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLO](https://img.shields.io/badge/YOLO-v8-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange.svg)

---

## ✨ Features

- 🎯 **Scratch Detection**: Identifies surface-level scratches and abrasions on belt surfaces
- 🔍 **Edge Damage Detection**: Detects irregularities and damage at belt boundaries  
- 🌓 **Day/Night Support**: Works with images captured under various lighting conditions
- 📊 **Structured Output**: Generates annotated images and JSON detection files
- ⚡ **Fast Processing**: Optimized pipeline for batch image processing

---

## 📁 Project Structure

```
conveyor-damage-detection/
├── 📂 data/
│   └── 📂 train/
│       ├── 📂 images/          # Input images (359 samples)
│       └── 📂 labels/          # Belt region annotations
├── 📂 models/
│   └── ⚙️ damage_config.json   # Detection parameters
├── 📂 runs/
│   └── 📂 detect/train/weights/
│       └── 🎯 best.pt          # Belt ROI model (6.2 MB)
├── 📂 outputs/                 # Detection results
│   ├── 🖼️ image_001.jpg        # Annotated image
│   └── 📄 image_001.json       # Detection data
├── ⚙️ belt.yaml                # Belt detection config
├── ⚙️ damage.yaml              # Damage classes config
├── 📝 train_belt.py            # Belt model training
├── 📝 train_damage.py          # Damage detection setup
├── 📝 pipeline.py              # Inference pipeline
└── 📋 requirements.txt         # Dependencies
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Navigate to project
cd conveyor-damage-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Damage Detection

```bash
python pipeline.py --image_dir data/train/images --output_dir outputs
```

---

## 📖 Usage Guide

### Training Pipeline

The system uses a two-stage approach for accurate damage detection:

#### Stage 1: Belt Region Detection
```bash
python train_belt.py
```
Trains a YOLOv8n model to precisely locate conveyor belt regions in images.

**Output**: `runs/detect/train/weights/best.pt`

#### Stage 2: Damage Detection Setup
```bash
python train_damage.py
```
Configures image processing parameters for scratch and edge damage detection.

**Output**: `models/damage_config.json`

### Inference Pipeline

```bash
python pipeline.py --image_dir <input_folder> --output_dir <output_folder>
```

**Arguments**:
- `--image_dir`: Path to folder containing input images
- `--output_dir`: Path to folder for saving results

---

## 📤 Output Examples

### Annotated Images

For each input image, the system generates an annotated version with bounding boxes:

| Color | Damage Type | Description |
|-------|-------------|-------------|
| 🔴 Red | `scratch` | Surface scratches and abrasions |
| 🟡 Yellow | `edge_damage` | Edge irregularities and damage |

**Example Output File**: `outputs/20260131_220303_792369.jpg`

### JSON Output Format

Each detection generates a structured JSON file with bounding box coordinates and damage classification:

**File**: `outputs/20260131_220303_792369.json`

```json
{
    "1": {
        "bbox_coordinates": [1070, 2143, 1128, 2160],
        "damage_type": "scratch"
    },
    "2": {
        "bbox_coordinates": [843, 0, 3080, 45],
        "damage_type": "edge_damage"
    },
    "3": {
        "bbox_coordinates": [2322, 1781, 2401, 1820],
        "damage_type": "scratch"
    }
}
```

**Field Description**:
- `bbox_coordinates`: [x_min, y_min, x_max, y_max] in pixels
- `damage_type`: Either `"scratch"` or `"edge_damage"`

---

## 🧠 Technical Approach

### Two-Stage Detection Pipeline

```
Input Image
    ↓
┌─────────────────┐
│  Belt Detection │  ← YOLOv8 Model (best.pt)
│   (Stage 1)     │
└─────────────────┘
    ↓
Cropped Belt Region
    ↓
┌─────────────────┐
│ Damage Detection│  ← Image Processing
│   (Stage 2)     │
└─────────────────┘
    ↓
Annotated Output + JSON
```

### Scratch Detection Algorithm

1. **Preprocessing**
   - Convert to grayscale
   - Apply CLAHE contrast enhancement (clipLimit=2.0, tileGridSize=(8,8))

2. **Edge Detection**
   - Canny edge detection (low=50, high=150)
   - Morphological operations (dilation + erosion) to connect edges

3. **Contour Analysis**
   - Filter by area (100-20000 pixels)
   - Aspect ratio filtering (>2 or <0.5 for elongated scratches)
   - Bounding box extraction

### Edge Damage Detection Algorithm

1. **Region Extraction**
   - Analyze top and bottom 30-pixel margins of cropped belt

2. **Edge Analysis**
   - Canny edge detection on margin regions
   - Edge response thresholding (>2000)

3. **Damage Classification**
   - Flag regions with significant edge irregularities

---

## 📦 Model Weights

| Model | File | Size | Description |
|-------|------|------|-------------|
| Belt ROI | `runs/detect/train/weights/best.pt` | 6.2 MB | Detects conveyor belt boundaries |
| Config | `models/damage_config.json` | <1 KB | Detection parameters |

**Training Details**:
- 359 training images
- 20 training epochs
- YOLOv8n architecture
- Input resolution: 640x640

---

## 🔬 Performance Evaluation

The system is evaluated using **mF1@0.5-0.95** (mean F1 score across IoU thresholds 0.50 to 0.95).

**Evaluation Methodology**:
- Bounding box matching via greedy IoU-based assignment
- True Positive: IoU ≥ threshold
- False Positive: IoU < threshold or duplicate detection
- False Negative: Unmatched ground truth

---

## 📋 Requirements

```
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
```

See `requirements.txt` for complete dependency list.

---

## 💡 Tips for Best Results

- Ensure images are clear and in focus
- Consistent lighting improves detection accuracy
- For custom belts, consider fine-tuning the belt detection model
- Adjust thresholds in `damage_config.json` for your specific use case

---

## 🤝 Support

For questions or issues, please refer to the code documentation or contact the development team.
# conveyor-damage-detection
