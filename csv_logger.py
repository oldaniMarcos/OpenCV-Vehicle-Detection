import json
from pathlib import Path
from collections import defaultdict

class CSVLogger:
  """
  Logs detection statistics and exports to JSON format.
  Tracks total vehicles, counts by lane (on crossing), and counts by final vehicle class.
  """

  def __init__(self, output_dir='./output'):
      """
      Initialize the CSV logger.

      Args:
        output_dir: Directory to save the JSON statistics file
      """
      self.output_dir = Path(output_dir)
      self.output_dir.mkdir(parents=True, exist_ok=True)
      
      # Track statistics
      self.tracked_vehicles = {}  # {tracker_id: class_name}
      self.finalized_vehicles = {}  # {tracker_id: class_name} - captured as finalized
      self.lane_counter_ref = None
      self.class_names = {}
      self.prev_finalized_ids = set()  # Track which vehicles were finalized in previous frame

  def update(self, detections, classifier):
    """
    Update statistics with current frame detections and capture finalized vehicles.

    Args:
      detections: supervision Detections object
      classifier: VehicleClassifier to check for newly finalized vehicles
    """
    if len(detections) == 0:
      self._capture_finalized_vehicles(classifier)
      return

    if detections.tracker_id is None:
      self._capture_finalized_vehicles(classifier)
      return

    # Track unique vehicles
    for tracker_id in detections.tracker_id:
      if tracker_id is not None:
        tracker_id = int(tracker_id)
        if tracker_id not in self.tracked_vehicles:
          self.tracked_vehicles[tracker_id] = None

    # Capture any newly finalized vehicles
    self._capture_finalized_vehicles(classifier)

  def _capture_finalized_vehicles(self, classifier):
    """
    Capture vehicles that just reached finalization.
    This prevents them from being lost when pruned later.
    """
    current_finalized_ids = set(classifier.vehicle_final_class.keys())
    
    # Find newly finalized vehicles (ones that weren't finalized before)
    newly_finalized = current_finalized_ids - self.prev_finalized_ids
    
    for tracker_id in newly_finalized:
      if tracker_id in classifier.vehicle_final_class:
        class_id = classifier.vehicle_final_class[tracker_id]
        class_name = self.class_names.get(class_id, f"class_{class_id}")
        self.finalized_vehicles[tracker_id] = class_name
    
    self.prev_finalized_ids = current_finalized_ids

  def set_class_names(self, class_names):
    """
    Set the class names mapping.

    Args:
      class_names: Dict or list mapping class_id to class_name
    """
    self.class_names = class_names

  def set_lane_counter(self, lane_counter):
    """
    Store reference to lane counter for reading final counts at export.

    Args:
      lane_counter: LaneCounter object with line zones
    """
    self.lane_counter_ref = lane_counter

  def finalize(self, classifier):
    """
    Finalize statistics by reading remaining classes from classifier.
    Finalized vehicles were already captured during processing.
    Call this after all frames have been processed but before export.

    Args:
      classifier: VehicleClassifier instance
    """
    # Process remaining tracked vehicles
    for tracker_id in self.tracked_vehicles.keys():
      # Skip if already finalized during processing
      if tracker_id in self.finalized_vehicles:
        self.tracked_vehicles[tracker_id] = self.finalized_vehicles[tracker_id]
        continue
      
      if tracker_id in classifier.vehicle_final_class:
        # Vehicle reached classification threshold
        class_id = classifier.vehicle_final_class[tracker_id]
        self.tracked_vehicles[tracker_id] = self.class_names.get(class_id, f"class_{class_id}")
      elif tracker_id in classifier.vehicle_votes:
        # Vehicle has votes but didn't reach threshold - use most common vote
        votes = classifier.vehicle_votes[tracker_id]
        if votes:
          most_common_class_id = votes.most_common(1)[0][0]
          self.tracked_vehicles[tracker_id] = self.class_names.get(most_common_class_id, f"class_{most_common_class_id}")
        else:
          self.tracked_vehicles[tracker_id] = "unclassified"
      elif tracker_id in self.finalized_vehicles:
        # Already have it from finalization capture
        self.tracked_vehicles[tracker_id] = self.finalized_vehicles[tracker_id]
      else:
        # Vehicle tracked but no classification data
        self.tracked_vehicles[tracker_id] = "unclassified"

  def export(self):
    """
    Export statistics to JSON file with lane crossing counts and final class counts.

    Returns:
      Path to the exported file
    """
    # Count vehicles by final class (including unclassified)
    class_counts = defaultdict(int)
    for vehicle_class in self.tracked_vehicles.values():
      if vehicle_class:
        class_counts[vehicle_class.lower()] += 1
      else:
        class_counts["unclassified"] += 1

    # Get lane crossing counts from line zones
    lane_counts = {
      "left": 0,
      "center": 0,
      "right": 0
    }
    
    if self.lane_counter_ref and hasattr(self.lane_counter_ref, 'line_zones'):
      lane_mapping = {
        "left lane": "left",
        "center lane": "center",
        "right lane": "right",
      }
      for lane_name, line_zone in self.lane_counter_ref.line_zones.items():
        if lane_name in lane_mapping:
          simplified_name = lane_mapping[lane_name]
          # Read cumulative crossing count from line zone
          lane_counts[simplified_name] = line_zone.in_count

    # Build statistics dictionary
    stats = {
      "total_vehicles": len(self.tracked_vehicles),
      "lanes": lane_counts,
      "classes": dict(class_counts)
    }

    # Save to JSON
    output_file = self.output_dir / 'statistics.json'
    with open(output_file, 'w') as f:
      json.dump(stats, f, indent=2)

    print(f"Statistics exported to: {output_file}")
    return output_file

  def get_summary(self):
    """Get a text summary of statistics."""
    class_counts = defaultdict(int)
    for vehicle_class in self.tracked_vehicles.values():
      if vehicle_class:
        class_counts[vehicle_class] += 1

    summary = f"""
Detection Statistics:
Total vehicles: {len(self.tracked_vehicles)}
Classes: {dict(class_counts)}
    """
    return summary
