# AeroGuard: HPC-Aware Digital Twin for Predictive Maintenance

## Overview
AeroGuard is a real-time Digital Twin application designed to estimate the Remaining Useful Life (RUL) of turbofan jet engines. By integrating physics-informed feature engineering with machine learning, the system moves beyond black-box predictions to provide interpretable diagnostics of engine health.

Built on the NASA C-MAPSS dataset, this project simulates a condition-based maintenance system. It visualizes high-stress zones (thermal and vibrational) directly on a 2D engine schematic, allowing operators to detect component degradation before catastrophic failure occurs.

## Key Features
* **Physics-Informed AI:** Utilizes specific degradation slopes (`fslope`) and Low-Frequency Energy Ratios (LF-ER) extracted via Fast Fourier Transform to model mechanical fatigue.
* **Accurate RUL Prediction:** Features an optimized XGBoost regressor that predicts failure cycles with high precision (RMSE < 24 cycles).
* **Digital Twin Visualization:** Dynamically overlays status alerts on the engine core and fan components based on live sensor thresholds.
* **Interactive Telemetry:** Allows users to replay historical test flights cycle-by-cycle to observe the evolution of component degradation.

## Technical Architecture

### 1. Data Ingestion & Processing
The system processes high-frequency sensor data from the NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset. Raw telemetry is cleaned and normalized using Min-Max scaling.

### 2. Feature Engineering (HPC-Aware)
To handle the noise inherent in raw sensor data, the pipeline implements High-Performance Computing techniques for feature extraction:
* **Rolling Window Statistics:** Calculates moving averages and standard deviations over 30-cycle windows to smooth transient noise.
* **Linear Degradation Trends:** Computes the slope of sensor trajectories to quantify the rate of component decay.
* **Signal Processing:** Applies Fourier Transforms to vibration sensors to isolate energy shifts indicative of mechanical faults.

### 3. Machine Learning Model
* **Algorithm:** XGBoost Regressor (Gradient Boosting).
* **Training:** Trained on historical run-to-failure trajectories.
* **Validation:** Validated using a hold-out test set to ensure generalization to unseen engines.

### 4. Visualization Layer
The frontend is built with Streamlit and Plotly, featuring a custom Pillow-based image processing layer that maps numerical risk scores to visual heatmaps on the engine schematic.

## Installation and Usage

### Prerequisites
* Python 3.8+
* pip (Python Package Manager)

### Setup
1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/aeroguard-digital-twin.git](https://github.com/your-username/aeroguard-digital-twin.git)
    cd aeroguard-digital-twin
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Run the application:
    ```bash
    streamlit run app.py
    ```

## Project Structure
* `app.py`: Main application entry point containing the dashboard logic.
* `aeroguard_brain.json`: Pre-trained XGBoost model file.
* `processed_test_data.csv`: Cleaned dataset with engineered features for simulation.
* `requirements.txt`: List of Python dependencies.
* `engine.jpg`: Schematic asset for the Digital Twin overlay.

## Performance
The model achieves a Root Mean Squared Error (RMSE) of approximately 23.5 cycles on the FD001 test set, effectively predicting failures well within the standard maintenance window for commercial aviation.

## License
This project is open-source and available under the MIT License.
