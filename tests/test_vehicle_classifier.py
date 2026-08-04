import numpy as np

from vehicle_classifier import VehicleClassifier


class FakeDetections:
  def __init__(self, class_id, tracker_id=None, confidence=None):
    self.class_id = np.asarray(class_id, dtype=np.int32)
    self.tracker_id = np.asarray(
      tracker_id if tracker_id is not None else [None] * len(class_id),
      dtype=object,
    )
    self.confidence = np.asarray(
      confidence if confidence is not None else [0.5] * len(class_id),
      dtype=np.float32,
    )

  def __len__(self):
    return len(self.class_id)


def test_process_returns_labels_for_empty_detections():
  classifier = VehicleClassifier(class_names=["car", "truck"], classification_votes=1)
  detections = FakeDetections([])

  labels = classifier.process(detections)

  assert labels == []
  assert not isinstance(labels, tuple)


def test_process_prunes_stale_tracker_state():
  classifier = VehicleClassifier(
    class_names=["car", "truck"],
    classification_votes=1,
    max_idle_frames=0,
  )

  first_frame = FakeDetections([0], tracker_id=[7], confidence=[0.9])
  classifier.process(first_frame)

  assert classifier.vehicle_final_class
  assert classifier.vehicle_final_confidence
  assert classifier.vehicle_best_confidence

  empty_frame = FakeDetections([])
  classifier.process(empty_frame)

  assert classifier.vehicle_final_class == {}
  assert classifier.vehicle_final_confidence == {}
  assert classifier.vehicle_best_confidence == {}
