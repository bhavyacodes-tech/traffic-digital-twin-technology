import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# COCO class IDs for vehicles: 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
CLASS_COLORS = {
    "Car": (0, 255, 255),        # Yellow/Gold
    "Motorcycle": (255, 200, 0), # Cyan
    "Bus": (0, 140, 255),        # Orange
    "Truck": (0, 0, 255)         # Red
}

class TrafficVisionEngine:
    def __init__(self, model_name="yolov8n.pt", num_lanes=3):
        """
        Initializes YOLOv8 with ByteTrack for multi-object tracking
        and maintains vehicle history for speed & lane queue estimation.
        """
        self.model = YOLO(model_name)
        self.num_lanes = num_lanes
        self.track_history = defaultdict(lambda: [])
        self.vehicle_speeds = {}
        self.cumulative_ids = set()

    def process_frame(self, frame, fps=30.0, ppm=8.5):
        """
        Processes a single video frame:
        - Detects & tracks vehicles using YOLOv8 + ByteTrack
        - Maps vehicles to Lane 1, Lane 2, or Lane 3
        - Estimates speed (km/h) and queue length (stopped/slow vehicles)
        """
        height, width, _ = frame.shape
        annotated = frame.copy()

        # Define lane boundaries across the road width
        lane_width = width // self.num_lanes
        lane_data = {
            i: {"count": 0, "queue": 0, "speeds": [], "vehicles": []}
            for i in range(1, self.num_lanes + 1)
        }
        class_counts = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0}

        # Draw Semi-Transparent Lane Dividers & Headers on Video
        for i in range(1, self.num_lanes):
            x_div = i * lane_width
            cv2.line(annotated, (x_div, int(height * 0.2)), (x_div, height), (255, 215, 0), 2, cv2.LINE_AA)

        # Run YOLOv8 + ByteTrack
        results = self.model.track(
            frame,
            persist=True,
            classes=list(VEHICLE_CLASSES.keys()),
            tracker="bytetrack.yaml",
            verbose=False
        )

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, cls_id in zip(boxes, track_ids, clss):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                v_type = VEHICLE_CLASSES.get(cls_id, "Car")

                self.cumulative_ids.add(track_id)
                class_counts[v_type] += 1

                # Determine which lane the vehicle belongs to
                lane_idx = min(max((cx // lane_width) + 1, 1), self.num_lanes)

                # Update trajectory history for speed estimation
                history = self.track_history[track_id]
                history.append((cx, cy))
                if len(history) > 15:
                    history.pop(0)

                # Calculate Speed in km/h using Euclidean pixel displacement
                if len(history) >= 5:
                    dx = history[-1][0] - history[0][0]
                    dy = history[-1][1] - history[0][1]
                    pixel_dist = np.hypot(dx, dy)
                    time_elapsed = len(history) / max(fps, 1.0)
                    speed_mps = (pixel_dist / ppm) / time_elapsed
                    # Apply realistic bounds & smoothing
                    raw_kmh = np.clip(speed_mps * 3.6, 4.0, 95.0)
                    prev_speed = self.vehicle_speeds.get(track_id, raw_kmh)
                    speed_kmh = 0.7 * prev_speed + 0.3 * raw_kmh
                else:
                    speed_kmh = self.vehicle_speeds.get(track_id, 32.0)

                self.vehicle_speeds[track_id] = speed_kmh

                # Update Lane Metrics
                lane_data[lane_idx]["count"] += 1
                lane_data[lane_idx]["speeds"].append(speed_kmh)

                # A vehicle is part of a "Queue" if its speed drops below 18 km/h or it's in the lower half
                if speed_kmh < 18.0 or cy > int(height * 0.55):
                    lane_data[lane_idx]["queue"] += 1

                # Normalized Y position (0.0 at top of road, 1.0 at intersection stop line)
                norm_y = round(cy / height, 3)
                lane_data[lane_idx]["vehicles"].append({
                    "id": track_id,
                    "type": v_type,
                    "speed": round(speed_kmh, 1),
                    "norm_y": norm_y
                })

                # Draw Bounding Box & Motion Trail
                color = CLASS_COLORS.get(v_type, (0, 255, 0))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"#{track_id} {v_type} {int(speed_kmh)}km/h"
                cv2.putText(
                    annotated, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                )

                # Draw tracking trail points
                if len(history) > 1:
                    pts = np.array(history, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated, [pts], isClosed=False, color=color, thickness=2)

        # Draw Lane Status Overlay Bar at Top of Video
        for i in range(1, self.num_lanes + 1):
            x_pos = (i - 1) * lane_width + 15
            cnt = lane_data[i]["count"]
            q_len = lane_data[i]["queue"]
            cv2.rectangle(annotated, (x_pos - 5, 10), (x_pos + 195, 65), (15, 23, 42), -1)
            cv2.putText(
                annotated, f"LANE {i} | Count: {cnt}", (x_pos, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2
            )
            cv2.putText(
                annotated, f"Queue: {q_len} veh", (x_pos, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        return annotated, lane_data, class_counts, len(self.cumulative_ids)