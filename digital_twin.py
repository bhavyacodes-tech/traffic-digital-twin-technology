import time
import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class RoadDigitalTwin:
    def __init__(self, road_name="NH-24 Smart Intersection", num_lanes=3, lane_capacity=12):
        self.road_name = road_name
        self.num_lanes = num_lanes
        self.lane_capacity = lane_capacity
        self.history = []
        self.start_time = time.time()

    def update_twin_state(self, lane_data, class_counts, total_unique_vehicles):
        """Updates the Digital Twin state with newly extracted CV parameters."""
        elapsed_sec = round(time.time() - self.start_time, 1)
        total_active = sum(lane_data[i]["count"] for i in range(1, self.num_lanes + 1))
        total_queue = sum(lane_data[i]["queue"] for i in range(1, self.num_lanes + 1))

        all_speeds = [s for i in range(1, self.num_lanes + 1) for s in lane_data[i]["speeds"]]
        avg_speed = round(float(np.mean(all_speeds)), 1) if all_speeds else 42.0

        lane_occupancy = {}
        for i in range(1, self.num_lanes + 1):
            occ = min(round((lane_data[i]["count"] / self.lane_capacity) * 100, 1), 100.0)
            lane_occupancy[i] = occ

        overall_occupancy = round(float(np.mean(list(lane_occupancy.values()))), 1)

        if overall_occupancy >= 65 or avg_speed < 18:
            congestion_level = "HIGH CONGESTION"
        elif overall_occupancy >= 35 or avg_speed < 30:
            congestion_level = "MODERATE FLOW"
        else:
            congestion_level = "FREE FLOW"

        snapshot = {
            "timestamp": time.strftime("%H:%M:%S"),
            "elapsed_sec": elapsed_sec,
            "total_active": total_active,
            "total_unique": total_unique_vehicles,
            "total_queue": total_queue,
            "avg_speed": avg_speed,
            "overall_occupancy": overall_occupancy,
            "congestion_level": congestion_level,
            "lane_1_count": lane_data[1]["count"],
            "lane_2_count": lane_data[2]["count"],
            "lane_3_count": lane_data[3]["count"],
            "lane_1_queue": lane_data[1]["queue"],
            "lane_2_queue": lane_data[2]["queue"],
            "lane_3_queue": lane_data[3]["queue"],
            "lane_occupancy": lane_occupancy,
            "lane_vehicles": {i: lane_data[i]["vehicles"] for i in range(1, self.num_lanes + 1)},
            "Car": class_counts.get("Car", 0),
            "Motorcycle": class_counts.get("Motorcycle", 0),
            "Bus": class_counts.get("Bus", 0),
            "Truck": class_counts.get("Truck", 0),
        }

        self.history.append(snapshot)
        if len(self.history) > 120:
            self.history.pop(0)

        return snapshot

    def forecast_short_term(self, horizon_steps=10):
        """Uses Linear Regression on recent Digital Twin states to forecast traffic buildup."""
        if len(self.history) < 4:
            current_val = self.history[-1]["total_active"] if self.history else 5
            return [current_val] * horizon_steps, "STABLE (Calibrating Sensor Stream...)"

        df = pd.DataFrame(self.history[-25:])
        X = np.arange(len(df)).reshape(-1, 1)
        y = df["total_active"].values

        model = LinearRegression()
        model.fit(X, y)

        future_X = np.arange(len(df), len(df) + horizon_steps).reshape(-1, 1)
        preds = np.clip(model.predict(future_X), 0, self.lane_capacity * self.num_lanes)
        preds = [round(float(p), 1) for p in preds]

        slope = model.coef_[0]
        if slope > 0.3 or preds[-1] > (self.lane_capacity * self.num_lanes * 0.65):
            trend = "WARNING: Rapid Queue Buildup Expected (+5 min)"
        elif slope > 0.08:
            trend = "MODERATE: Gradual Traffic Increase"
        elif slope < -0.12:
            trend = "CLEARING: Congestion Dissipating"
        else:
            trend = "NORMAL: Steady Traffic Flow Expected"

        return preds, trend

    def render_twin_canvas_rgb(self, current_state, width=840, height=480):
        """
        Renders an ultra-fast, zero-lag 2D Schematic Digital Twin image (RGB)
        so it updates at the exact same millisecond as the CCTV video frame.
        """
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (15, 23, 42)  # Dark slate background (RGB)

        margin_top = 65
        margin_bottom = 25
        road_h = height - margin_top - margin_bottom
        lane_w = width // self.num_lanes

        # Title Banner
        cv2.rectangle(canvas, (0, 0), (width, 50), (30, 41, 59), -1)
        cv2.putText(
            canvas, f"LIVE 2D DIGITAL TWIN: {self.road_name.upper()}",
            (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 204, 21), 2, cv2.LINE_AA
        )

        # Stop line at bottom of intersection
        stop_y = height - margin_bottom - 15

        for i in range(1, self.num_lanes + 1):
            x0 = (i - 1) * lane_w + 10
            x1 = i * lane_w - 10
            occ = current_state["lane_occupancy"][i]
            q_cnt = current_state[f"lane_{i}_queue"]
            v_cnt = current_state[f"lane_{i}_count"]

            # Dynamic Lane Background Color based on Congestion State
            if occ >= 65:
                lane_bg = (75, 25, 35)      # High Congestion Red Tint
                status_col = (239, 68, 68)
            elif occ >= 35:
                lane_bg = (70, 55, 20)      # Moderate Amber Tint
                status_col = (245, 158, 11)
            else:
                lane_bg = (20, 55, 45)      # Free Flow Green Tint
                status_col = (16, 185, 129)

            cv2.rectangle(canvas, (x0, margin_top), (x1, stop_y), lane_bg, -1)
            cv2.rectangle(canvas, (x0, margin_top), (x1, stop_y), status_col, 2)

            # Draw Queue Zone Fill from the Stop Line upward
            if q_cnt > 0:
                q_height = int(min((q_cnt / self.lane_capacity) * road_h, road_h - 10))
                cv2.rectangle(canvas, (x0 + 4, stop_y - q_height), (x1 - 4, stop_y), (120, 30, 40), -1)

            # Draw Dashed Center Guide Line inside each lane
            mid_x = (x0 + x1) // 2
            for y_dash in range(margin_top + 50, stop_y - 20, 35):
                cv2.line(canvas, (mid_x, y_dash), (mid_x, y_dash + 15), (100, 116, 139), 1)

            # Lane Header Box
            cv2.rectangle(canvas, (x0, margin_top), (x1, margin_top + 48), (15, 23, 42), -1)
            cv2.putText(
                canvas, f"LANE {i} | Occ: {int(occ)}%", (x0 + 10, margin_top + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_col, 2, cv2.LINE_AA
            )
            cv2.putText(
                canvas, f"Vehicles: {v_cnt} | Queue: {q_cnt}", (x0 + 10, margin_top + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (226, 232, 240), 1, cv2.LINE_AA
            )

            # Draw Synchronized Virtual Vehicles inside the Digital Twin
            vehicles = current_state["lane_vehicles"].get(i, [])
            for idx, v in enumerate(vehicles):
                # Spread vehicles horizontally slightly so boxes never overlap
                offset_x = int(np.sin(v["id"] * 1.7 + idx) * ((x1 - x0) * 0.25))
                vx = mid_x + offset_x
                vy = int(margin_top + 65 + v["norm_y"] * (road_h - 90))
                vy = int(np.clip(vy, margin_top + 60, stop_y - 22))

                # Color vehicle red if queued/slow (<18 km/h), cyan if moving freely
                v_col = (239, 68, 68) if v["speed"] < 18.0 else (56, 189, 248)
                cv2.rectangle(canvas, (vx - 22, vy - 14), (vx + 22, vy + 14), v_col, -1)
                cv2.rectangle(canvas, (vx - 22, vy - 14), (vx + 22, vy + 14), (255, 255, 255), 1)

                cv2.putText(
                    canvas, f"#{v['id']}", (vx - 18, vy - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (15, 23, 42), 2, cv2.LINE_AA
                )
                cv2.putText(
                    canvas, f"{int(v['speed'])}kph", (vx - 20, vy + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA
                )

        # Draw Intersection Stop Line at Bottom
        cv2.line(canvas, (10, stop_y), (width - 10, stop_y), (239, 68, 68), 4)
        cv2.putText(
            canvas, "INTERSECTION SIGNAL STOP LINE", (width // 2 - 130, height - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (248, 113, 113), 1, cv2.LINE_AA
        )

        return canvas


def generate_sumo_simulation_step(step_idx, num_lanes=3, width=840, height=480):
    """
    SUMO-Style Synthetic Traffic Network Generator (Objective 7).
    Returns both the Digital Twin telemetry AND a live simulated CCTV camera frame!
    """
    lane_data = {}
    wave = np.sin(step_idx / 7.0) * 0.5 + 0.5
    class_counts = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0}

    # Create a Synthetic CCTV Highway Frame for the left panel
    sim_frame = np.zeros((height, width, 3), dtype=np.uint8)
    sim_frame[:] = (35, 40, 48)  # Asphalt gray
    lane_w = width // num_lanes

    for i in range(1, num_lanes):
        cv2.line(sim_frame, (i * lane_w, 50), (i * lane_w, height), (250, 204, 21), 2)

    cv2.rectangle(sim_frame, (0, 0), (width, 50), (15, 23, 42), -1)
    cv2.putText(
        sim_frame, "SUMO SYNTHETIC TRAFFIC NETWORK STREAM (LIVE SIMULATION)",
        (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (56, 189, 248), 2, cv2.LINE_AA
    )

    np.random.seed(step_idx + 42)
    for lane in range(1, num_lanes + 1):
        count = int(np.clip(3 + int(wave * 6) + (lane % 2), 2, 10))
        speeds = []
        vehicles = []
        queue = 0
        x_center = (lane - 1) * lane_w + lane_w // 2

        for v in range(count):
            vid = 100 + lane * 20 + v
            v_type = ["Car", "Motorcycle", "Bus", "Truck"][(vid + v) % 4]
            class_counts[v_type] += 1

            spd = round(float(np.clip(54 - count * 4.2 + (v % 3) * 3, 8.0, 62.0)), 1)
            norm_y = round(float(((step_idx * 0.06 + v * (0.85 / count)) % 0.85) + 0.08), 3)

            if spd < 18.0 or norm_y > 0.68:
                queue += 1

            speeds.append(spd)
            vehicles.append({"id": vid, "type": v_type, "speed": spd, "norm_y": norm_y})

            # Draw Simulated Vehicle + YOLO Bounding Box on the Left Panel
            vx = x_center + int(np.sin(vid) * (lane_w * 0.22))
            vy = int(60 + norm_y * (height - 90))
            cv2.rectangle(sim_frame, (vx - 26, vy - 18), (vx + 26, vy + 18), (250, 204, 21), 2)
            cv2.rectangle(sim_frame, (vx - 22, vy - 14), (vx + 22, vy + 14), (70, 130, 180), -1)
            cv2.putText(
                sim_frame, f"#{vid} {v_type} {int(spd)}km/h", (vx - 35, vy - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA
            )

        lane_data[lane] = {"count": count, "queue": queue, "speeds": speeds, "vehicles": vehicles}

    total_unique = 45 + step_idx * 2
    return lane_data, class_counts, total_unique, sim_frame