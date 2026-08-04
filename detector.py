from ultralytics import YOLO
import supervision as sv
import numpy as np

class Detector:
  
  def __init__(
    self, 
    model_path, 
    tracker_path, 
    confidence_threshold, 
    vehicle_classes, 
    device
  ):
    
    if device == 'cpu':
      print("Warning: Using CPU for inference. This may be slow. Consider using a GPU for better performance.")
    
    self.model = YOLO(model_path)
    self.model.to(device)
    
    self.tracker_path = tracker_path
    self.confidence_threshold = confidence_threshold
    self.vehicle_classes = vehicle_classes
  
  @property
  def class_names(self):
    return self.model.names
  
  def detect(self, frame):
    
    results = self.model.track(
      frame,
      persist=True,
      tracker=self.tracker_path,
      verbose=False,
      conf=self.confidence_threshold,
    )[0]

    detections = sv.Detections.from_ultralytics(results)

    detections = detections[
      np.isin(
        detections.class_id,
        self.vehicle_classes
      )
    ]

    return detections
