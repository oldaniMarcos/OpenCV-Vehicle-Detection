from collections import defaultdict, Counter
import numpy as np


class VehicleClassifier:

  def __init__(
    self,
    class_names,
    classification_votes,
    max_idle_frames=10,
  ):

    self.class_names = class_names
    self.classification_votes = classification_votes
    self.max_idle_frames = max_idle_frames

    self.vehicle_votes = defaultdict(Counter)
    self.vehicle_final_class = {}
    self.vehicle_best_confidence = {}
    self.vehicle_final_confidence = {}
    self.vehicle_idle_frames = {}

  def process(self, detections):

    labels = []
    stable_class_ids = []

    if len(detections) == 0:
      self._prune_idle_tracker_state()
      return labels
      
    if detections.tracker_id is None:
      
      labels = [
        self.class_names[class_id] for class_id in detections.class_id
      ]
      
      return labels

    active_tracker_ids = set()

    for class_id, tracker_id, confidence in zip(
      detections.class_id,
      detections.tracker_id,
      detections.confidence
    ):

      if tracker_id is None:

        labels.append(self.class_names[class_id])
        stable_class_ids.append(class_id)
        continue

      tracker_id = int(tracker_id)
      active_tracker_ids.add(tracker_id)
      self.vehicle_idle_frames[tracker_id] = 0

      if tracker_id in self.vehicle_final_class:

        stable_class = self.vehicle_final_class[tracker_id]
        confidence = self.vehicle_final_confidence[tracker_id]

      else:

        self.vehicle_best_confidence[tracker_id] = max(
          self.vehicle_best_confidence.get(tracker_id, 0.0),
          confidence
        )

        self.vehicle_votes[tracker_id][int(class_id)] += 1

        total_votes = sum(
          self.vehicle_votes[tracker_id].values()
        )

        stable_class = self.vehicle_votes[
          tracker_id
        ].most_common(1)[0][0]

        if total_votes >= self.classification_votes:

          self.vehicle_final_class[tracker_id] = stable_class

          self.vehicle_final_confidence[tracker_id] = (
            self.vehicle_best_confidence[tracker_id]
          )

          del self.vehicle_votes[tracker_id]

          stable_class = self.vehicle_final_class[tracker_id]
          confidence = self.vehicle_final_confidence[tracker_id]

      stable_class_ids.append(stable_class)

      labels.append(
        f"{self.class_names[stable_class]} "
        f"#{tracker_id} ({confidence:.2f})"
      )

    self._prune_idle_tracker_state(active_tracker_ids)

    detections.class_id = np.asarray(
      stable_class_ids,
      dtype=detections.class_id.dtype
    )

    return labels

  def _prune_idle_tracker_state(self, active_tracker_ids=None):

    if active_tracker_ids is None:
      active_tracker_ids = set()

    for tracker_id in list(self.vehicle_idle_frames):
      if tracker_id in active_tracker_ids:
        self.vehicle_idle_frames[tracker_id] = 0
        continue
        
      self.vehicle_idle_frames[tracker_id] += 1

      if self.vehicle_idle_frames[tracker_id] >= self.max_idle_frames:
        self.vehicle_votes.pop(tracker_id, None)
        self.vehicle_final_class.pop(tracker_id, None)
        self.vehicle_best_confidence.pop(tracker_id, None)
        self.vehicle_final_confidence.pop(tracker_id, None)
        self.vehicle_idle_frames.pop(tracker_id, None)