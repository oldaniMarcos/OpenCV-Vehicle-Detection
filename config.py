import torch

MODEL_PATH = './models/yolo11m.pt'
VIDEO_PATH = './media/sample.mp4'
MASK_PATH = './masks/mask2.png'
TRACKER_PATH = './trackers/bytetrack.yaml'
OUTPUT_PATH = './output/output.mp4'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WINDOW_NAME = "Vehicle Detection and Tracking"

CONFIDENCE_THRESHOLD = 0.5
VEHICLE_CLASSES = [2, 3, 5, 7]
CLASSIFICATION_VOTES = 20

CLASS_COLORS = {
  2: 0,  # Car
  3: 1,  # Motorcycle
  5: 2,  # Bus
  7: 3,  # Truck
}