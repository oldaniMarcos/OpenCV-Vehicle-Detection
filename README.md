# Vehicle Detection and Tracker with OpenCV, YOLO

The following project uses OpenCV in combination with different YOLO11 models with the goal of detecting, classifying, and counting vehicles.

## Features

- **Real-time Vehicle Detection**: Uses YOLO11 models for fast and accurate vehicle detection
- **Multi-model Support**: Choose between 5 model sizes (from nano to extra-large)
- **Vehicle Classification**: Voting-based classification system to accurately identify vehicle types (car, truck, bus, motorcycle)
- **Multi-object Tracking**: ByteTrack integration for consistent vehicle tracking across frames
- **Lane Detection**: Monitors vehicle crossings in three lanes (left, center, right)
- **ROI Masking**: Uses region-of-interest masking to focus on specific areas
- **Statistics Export**: Generates JSON reports with vehicle counts by class and lane crossings
- **Optional Video Output**: Save processed video with detections and annotations

## Initial Setup

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (optional, for GPU acceleration)

### Installation

1. Clone or download this repository

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Download YOLO11 models (optional, they will auto-download on first run)

## Project Structure

```
├── main.py                # Main script to run detection and tracking
├── config.py              # Configuration settings and paths
├── detector.py            # YOLO detection and tracking logic
├── vehicle_classifier.py  # Vehicle classification with voting system
├── csv_logger.py          # Statistics logging and export
├── visualization.py       # Annotation and drawing functions
├── utils.py               # Utility functions
├── tests/                 # Test files
├── models/                # Directory for YOLO model weights (auto-populated)
├── masks/                 # ROI mask images
├── media/                 # Input video files
├── output/                # Output directory for videos and statistics
└── trackers/              # ByteTrack configuration
```

## Configuration

Edit `config.py` to customize:

- `VIDEO_PATH`: Path to input video file
- `MASK_PATH`: Path to ROI mask image
- `CONFIDENCE_THRESHOLD`: Detection confidence threshold (0.0-1.0)
- `CLASSIFICATION_VOTES`: Number of frames needed to finalize vehicle classification
- `VEHICLE_CLASSES`: Which YOLO classes to detect (default: cars, motorcycles, buses, trucks)
- `DEVICE`: GPU ('cuda') or CPU ('cpu')

## Usage

### Basic Usage

Run with default settings (medium YOLO11 model and no video output):

```bash
python main.py
```

### Command-line Arguments

#### Model Selection (`--model`)

Select a different model size:

```bash
python main.py --model n   # Nano (fastest, least accurate)
python main.py --model s   # Small
python main.py --model m   # Medium (default)
python main.py --model l   # Large
python main.py --model x   # Extra-large (slowest, most accurate)
```

#### Video Output (`--o`)

Save the processed video with annotations:

```bash
python main.py --o  # Saves output to configured OUTPUT_PATH
```

## Output

After running the script, you'll find:

- **`output/statistics.json`**: Detection statistics
- **`output/output.mp4`**: Annotated video with bounding boxes, class labels, FPS, and lane indicators (if `--o` flag is used)

## How It Works

1. **Detection**: YOLO11 model detects vehicles in each frame
2. **Tracking**: ByteTrack maintains consistent IDs across frames
3. **Classification**: Voting system classifies vehicles after multiple detections
4. **Lane Counting**: Line zones track vehicle crossings
5. **Export**: Statistics are compiled and saved to JSON

## Requirements

See `requirements.txt`
