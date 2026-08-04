import time
import cv2
import supervision as sv
import numpy as np
from config import CLASS_COLORS

palette = sv.ColorPalette.from_hex([
  "#BE0000",  # 0 -> Car
  "#7600D6",  # 1 -> Motorcycle
  "#A8A800",  # 2 -> Bus
  "#0000E0",  # 3 -> Truck
])

bounding_box_annotator = sv.BoxAnnotator(
  color=palette,
)

label_annotator = sv.LabelAnnotator(
  color=palette,
)


class FPSCounter:

  def __init__(self):
    self.prev_time = time.perf_counter()
    self.fps = 0.0

  def draw(self, frame):
    current_time = time.perf_counter()
    instant_fps = 1.0 / (current_time - self.prev_time)
    self.prev_time = current_time
    self.fps = 0.9 * self.fps + 0.1 * instant_fps

    cv2.putText(
      frame,
      f"FPS: {self.fps:.1f}",
      (10, 30),
      cv2.FONT_HERSHEY_SIMPLEX,
      1,
      (0, 255, 0),
      2,
      cv2.LINE_AA,
    )

    return frame

class LaneCounter:

  def __init__(self, frame_width, frame_height):
    self.frame_width = frame_width
    self.frame_height = frame_height

    self.line_zones = {
      "left lane": sv.LineZone(
        start=sv.Point(557, 86),
        end=sv.Point(641, 107),
        triggering_anchors=[sv.Position.BOTTOM_CENTER],
      ),
      "center lane": sv.LineZone(
        start=sv.Point(727, 73),
        end=sv.Point(947, 138),
        triggering_anchors=[sv.Position.BOTTOM_CENTER],
      ),
      "right lane": sv.LineZone(
        start=sv.Point(992, 647),
        end=sv.Point(413, 575),
        triggering_anchors=[sv.Position.BOTTOM_CENTER],
      ),
    }

    self.annotator = sv.LineZoneAnnotator(
      thickness=1,
      color = sv.Color.GREEN,
      text_thickness=1,
      text_scale=0.7,
      display_in_count=False,
      display_out_count=False,
    )

  def update_counts(self, detections):

    if len(detections) == 0:
      return

    if detections.tracker_id is None:
      return

    for line_zone in self.line_zones.values():
      line_zone.trigger(detections)

  def draw(self, frame):

    for lane_name, line_zone in self.line_zones.items():

      self.annotator.annotate(
        frame=frame,
        line_counter=line_zone,
      )

      cv2.putText(
        frame,
        f"{lane_name.title()}: {line_zone.in_count}",
        (10, 80 + 30 * list(self.line_zones.keys()).index(lane_name)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
      )

    return frame

def draw_detections(frame, detections, labels):

  custom_color_lookup = np.array(
    [
      CLASS_COLORS[int(class_id)]
      for class_id in detections.class_id
    ],
    dtype=np.int32,
  )

  frame = bounding_box_annotator.annotate(
    scene=frame,
    detections=detections,
    custom_color_lookup=custom_color_lookup,
  )

  frame = label_annotator.annotate(
    scene=frame,
    detections=detections,
    labels=labels,
    custom_color_lookup=custom_color_lookup,
  )

  return frame

