import argparse
import cv2
from pathlib import Path

from config import (
  MODEL_PATH,
  VIDEO_PATH,
  TRACKER_PATH,
  MASK_PATH,
  OUTPUT_PATH,
  CONFIDENCE_THRESHOLD,
  CLASSIFICATION_VOTES,
  VEHICLE_CLASSES,
  DEVICE,
  WINDOW_NAME
)
from visualization import draw_detections, FPSCounter, LaneCounter
from utils import load_mask, apply_mask, mouse_callback
from detector import Detector
from vehicle_classifier import VehicleClassifier
from csv_logger import CSVLogger

parser = argparse.ArgumentParser(description='Vehicle detection and tracking')
parser.add_argument('--o', action='store_true', help='Save output video (default: False)')
args = parser.parse_args()

detector = Detector(
  model_path=MODEL_PATH,
  tracker_path=TRACKER_PATH,
  confidence_threshold=CONFIDENCE_THRESHOLD,
  vehicle_classes=VEHICLE_CLASSES,
  device=DEVICE
)

classifier = VehicleClassifier(
  class_names=detector.class_names,
  classification_votes=CLASSIFICATION_VOTES
)
fps_counter = FPSCounter()
csv_logger = CSVLogger(output_dir=Path(OUTPUT_PATH).parent)
csv_logger.set_class_names(detector.class_names)
csv_logger.set_lane_counter(None)

cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
lane_counter = LaneCounter(width, height)
csv_logger.set_lane_counter(lane_counter)
if not cap.isOpened():
  raise Exception("Error opening video file.")
mask = load_mask(cap, MASK_PATH)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

writer = None
if args.o:
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
  writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
  
  if not writer.isOpened():
    raise RuntimeError(f"Could not create output video: {OUTPUT_PATH}")
  else:
    print(f"Saving output video to: {OUTPUT_PATH}")
else:
  print("Video output disabled (use --o to enable)")

while True:
  ret, frame = cap.read()
  
  if not ret:
    break
  
  masked_frame = apply_mask(frame, mask)
  
  detections = detector.detect(masked_frame)

  labels = classifier.process(detections)
  
  # Update statistics logger with unique vehicles and capture finalized classes
  csv_logger.update(detections, classifier)
  
  draw_detections(frame=frame, detections=detections, labels=labels)
  lane_counter.update_counts(detections)
  
  fps_counter.draw(frame)
  lane_counter.draw(frame)
  
  if writer is not None:
    writer.write(frame)

  cv2.namedWindow(WINDOW_NAME)
  # Uncomment the following line to enable mouse callback
  # cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
  cv2.imshow(WINDOW_NAME, frame)
  
  key = cv2.waitKey(1) & 0xFF
  if key == 27:
    break

cap.release()
if writer is not None:
  writer.release()
cv2.destroyAllWindows()

# Finalize statistics with final classes and export
csv_logger.finalize(classifier)
print(csv_logger.get_summary())
csv_logger.export()