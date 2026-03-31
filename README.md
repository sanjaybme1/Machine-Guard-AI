# MachineGuard AI Dashboard

An autonomous predictive maintenance system for industrial machinery. Built with Streamlit, Plotly, and NVIDIA NVIDIA NIM LLMs.

## Features
- **Real-time Sensor Monitoring**: Track Temperature, Vibration, and Oil Pressure.
- **AI Diagnostics**: Autonomous ReAct loop for detecting faults (Bearing, Cavitation, Overheating).
- **Health & RUL Forecasting**: Estimate Remaining Useful Life (RUL) and health index.
- **Anomaly Heatmaps**: Visual identification of abnormal signal patterns.

## Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set your NVIDIA API Key in the settings tab or environment variables.
3. Run the application:
   ```bash
   streamlit run app.py
   ```
