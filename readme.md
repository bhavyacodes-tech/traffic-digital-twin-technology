# 🚦 Traffic Digital Twin using Computer Vision & Short-Term Forecasting

A smart-city traffic monitoring system that transforms standard CCTV traffic footage into a real-time **2D Schematic Digital Twin** with **Short-Term Congestion Forecasting**.

## 📌 Key Features
1. **Computer Vision Detection & Tracking:** Uses **YOLOv8 (`yolov8n`)** and **ByteTrack** to detect and track Cars, Motorcycles, Buses, and Trucks across video frames.
2. **Traffic Parameter Extraction:** Automatically computes lane-wise vehicle counts, estimated speed (km/h), lane occupancy (%), and intersection queue length.
3. **Real-Time 2D Digital Twin:** Dynamically updates a virtual 3-lane schematic road model reflecting live vehicle positions, queue buildup, and congestion states (*Free Flow*, *Moderate*, *High Congestion*).
4. **Short-Term Congestion Forecasting:** Uses time-series linear regression on historical Digital Twin states to predict traffic buildup over the next 5 minutes.
5. **SUMO-Style Synthetic Simulation Mode:** Includes a built-in traffic wave generator to test the Digital Twin without live camera feeds.

## 🛠️ Tech Stack
- **Computer Vision:** OpenCV, Ultralytics YOLOv8, ByteTrack
- **Digital Twin & Forecasting:** Python, NumPy, Pandas, Scikit-Learn, Plotly
- **Web Dashboard:** Streamlit

## 🚀 How to Run Locally
```bash
pip install -r requirements.txt
python -m streamlit run app.py