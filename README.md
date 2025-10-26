
# ✈️ AI-Based Flight Safety & Anomaly Detection System

A real-time machine learning dashboard for detecting flight anomalies and monitoring pilot performance using ensemble ML models.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://flight-anomaly-detection-monitoring.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Dataset](#dataset)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

This project implements an **AI-powered flight safety monitoring system** that analyzes flight parameters in real-time to detect four critical types of anomalies:

- **Stall conditions** (low speed, high angle of attack)
- **Overspeed events** (excessive aircraft velocity)
- **Hard landings** (high vertical speed near ground)
- **Abrupt maneuvers** (excessive roll/pitch rates)

The system uses **Random Forest** and **XGBoost** ensemble models trained on 50,000+ synthetic flight records to achieve **93-95% accuracy** in anomaly detection.

---

## ✨ Features

### Core Capabilities
- ✅ **Real-time anomaly detection** with confidence scoring
- ✅ **Interactive dashboard** with live flight monitoring
- ✅ **Multiple visualization modes**: Live monitoring, data analysis, raw data view
- ✅ **File upload support** for custom flight data (CSV format)
- ✅ **Downloadable results** with predictions and confidence scores
- ✅ **Dynamic record filtering** (auto-adjusts to dataset size)
- ✅ **Pilot performance summary** with critical metrics

### Technical Features
- 🔬 Ensemble ML models (Random Forest + XGBoost)
- 📊 Interactive Plotly visualizations
- 🎨 Aviation-themed dark cockpit UI
- 📈 Real-time status monitoring with visual alerts
- 💾 Robust handling of large datasets (50k+ rows)

---

## 🎥 Demo

**Live App:** [https://flight-anomaly-detection-monitoring.streamlit.app](https://flight-anomaly-detection-monitoring.streamlit.app)

Try uploading sample CSV files to see anomaly detection in action!

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Setup

1. **Clone the repository**
```
git clone https://github.com/YOUR-USERNAME/flight-safety-dashboard.git
cd flight-safety-dashboard
```

2. **Create virtual environment** (recommended)
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```
pip install -r requirements.txt
```

4. **Run the app**
```
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📖 Usage

### Running Locally
```
streamlit run app.py
```

### Using the Dashboard

1. **Upload CSV File**: Click "Upload Flight Data CSV" in the sidebar
2. **Adjust Settings**: 
   - Set monitoring speed (0.1 - 2 seconds)
   - Choose number of records to analyze (auto-adjusts to file size)
3. **View Results**:
   - **Live Monitoring Tab**: Real-time anomaly detection with parameter display
   - **Data Analysis Tab**: Performance metrics, charts, and anomaly breakdown
   - **Raw Data Tab**: Complete dataset with predictions and confidence scores
4. **Download Results**: Export analyzed data with predictions as CSV

### Sample CSV Format
Your CSV should contain these columns:
```
speed_knots, altitude_ft, throttle_pct, aoa_deg, pitch_deg, roll_deg, vertical_speed_fpm, g_force
```

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 93-95% | 0.94 | 0.93 | 0.93 |
| XGBoost | 93-95% | 0.95 | 0.93 | 0.94 |

### Model Training
- **Dataset Size**: 50,000 synthetic flight records
- **Train/Test Split**: 80/20
- **Features**: 8 flight parameters
- **Target Classes**: Binary (Normal/Anomaly) + 4 anomaly types

---

## 📁 Dataset

### Synthetic Flight Data
The model is trained on synthetic data generated to simulate realistic flight scenarios:

- **Normal Operations** (95%): Standard cruise, climb, descent, landing
- **Anomalies** (5%): 
  - Stall conditions
  - Overspeed events
  - Hard landings
  - Abrupt turns

### Data Generation
Run `generate_all_test_files.py` to create test datasets:
```
python generate_all_test_files.py
```

This creates 5 test files:
- `test_all_normal.csv` (120 normal flights)
- `test_all_anomalies.csv` (120 anomalies)
- `test_mixed.csv` (200 mixed records)
- `test_edge_cases.csv` (150 edge scenarios)
- `test_realtime.csv` (120 complete flight simulation)

---

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **ML Models**: scikit-learn (Random Forest), XGBoost
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Deployment**: Streamlit Community Cloud
- **Version Control**: Git/GitHub

---

## 📂 Project Structure

```
flight-safety-dashboard/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── best_model.pkl             # Trained ML model
├── scaler.pkl                 # Feature scaler
├── feature_names.pkl          # Feature column names
├── ai_flight_safety.csv       # Default sample dataset
├── generate_all_test_files.py # Test data generator
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

---

## 📸 Screenshots

### Dashboard Overview
![Dashboard](https://via.placeholder.com/800x400?text=Add+Your+Screenshot+Here)

### Real-Time Monitoring
![Live Monitor](https://via.placeholder.com/800x400?text=Add+Your+Screenshot+Here)

### Data Analysis
![Analysis](https://via.placeholder.com/800x400?text=Add+Your+Screenshot+Here)

---

## 🔮 Future Enhancements

- [ ] Multi-class anomaly prediction (simultaneous detection)
- [ ] Historical trend analysis and reporting
- [ ] Integration with real aircraft data feeds
- [ ] Predictive maintenance alerts
- [ ] Multi-aircraft fleet monitoring
- [ ] Advanced pilot performance scoring
- [ ] Export PDF reports
- [ ] User authentication and role management

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

