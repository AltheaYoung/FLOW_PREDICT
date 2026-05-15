# ✈️ FLOW_PREDICT: Flight Flow Prediction & Intelligent Recovery System
> **Intelligent Flight Situation Reasoning and Collaborative Recovery Based on Multi-Source Data**

---

## 📖 Overview
**FLOW_PREDICT** is an end-to-end civil aviation intelligent system that integrates:
- 24-hour flight flow forecasting
- Real-time operational situation deduction
- Aircraft-crew-passenger collaborative recovery

It is designed to solve frequent flight disruptions caused by severe weather, air traffic control, and capacity constraints, helping airlines reduce delays, lower operational costs, and improve passenger experience.

---

## 🧩 Core System Architecture
The system consists of three highly coupled core modules:

### 1. Flight Flow Prediction Model
- **Architecture**: `Seq2Seq + Bi-LSTM + Attention Mechanism`
- **Input**: Historical flight data, METAR weather reports, weather forecasts
- **Key Innovations**:
  - Dynamic time-slot slicing
  - Separate modeling for busy/non-busy periods
  - Error-driven weighted fusion
- **Performance**:
  - Busy-hour average deviation: **6.04%**
  - Peak prediction error controlled within **10.6%**

### 2. 24-Hour Flight Operation Deduction
- **Method**: Discrete Event Simulation (DES)
- **Time Precision**: Minute-level
- **Key Functions**:
  - Simulate flight queuing, departure, and congestion evolution
  - Support "what-if" scenario analysis
  - Output predicted departure times and congestion reports

### 3. Multi-Resource Collaborative Recovery Optimization
- **Algorithm**: Mixed Integer Programming (MIP)
- **Solver**: Gurobi
- **Optimization Objectives**:
  - Minimize delayed flights
  - Minimize passenger delay cost
  - Minimize crew duty violations
- **Key Constraints**:
  - Hourly/departure capacity limits
  - Aircraft turnaround time
  - Crew duty time limits
  - International flight delay restrictions

---

## 📊 Performance & Results
- Delayed flights reduced by up to **47.55%**
- Average delay duration reduced by **24.46%**
- Passenger delay cost reduced by **24.98%**
- Solution time: Most cases within **20 seconds**
- Supports large-scale airport real-time dynamic scheduling

---

## 🏆 Awards
- National Third Prize, 19th *Challenge Cup* National College Student Academic Science and Technology Competition
- National Third Prize, 20th *Tsinghua IE Sword Competition*
- Second Prize, Sichuan University Robotics & AI Competition 2026

---

## 🛠️ Tech Stack
-Languages & Frameworks: Python, PyTorch
-Libraries: NumPy, Pandas
-Models & Methods: Bi-LSTM, Seq2Seq, Attention, Discrete Event Simulation (DES), Mixed Integer Programming (MIP)
-Solver: Gurobi

## 📂 Project Structure
```text
FLOW_PREDICT/
├── config/             # Configuration files
├── data_artifacts/     # Data, logs, and output artifacts
├── scripts/            # Preprocessing & evaluation scripts
├── src/                # Core model implementation
│   └── train_out.py    # Training & inference for flow prediction
├── README.md
└── requirements.txt


