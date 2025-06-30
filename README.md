# 🚦 NC Traffic Forecasting - Machine Learning Pipeline

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Nightly-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

A comprehensive machine learning and deep learning pipeline for forecasting North Carolina highway traffic volumes. This project demonstrates end-to-end ML workflows including data generation, feature engineering, traditional ML models, LSTM deep learning, evaluation, and interactive visualization.

## 📊 Project Overview

This project forecasts Average Annual Daily Traffic (AADT) volumes for major North Carolina highway segments using multiple machine learning approaches:

- **Traditional ML Models**: Linear Regression, Random Forest, Gradient Boosting
- **Deep Learning**: LSTM (Long Short-Term Memory) Neural Networks
- **Feature Engineering**: 29 engineered features including temporal, lag, and rolling statistics
- **Interactive Dashboard**: Streamlit-based visualization interface
- **Comprehensive Evaluation**: Multiple performance metrics and model comparison

## 🏆 Performance Results

| Model | RMSE | MAE | MAPE | R² | Directional Accuracy |
|-------|------|-----|------|----|---------------------|
| **Random Forest** 🥇 | 4,895.83 | 3,606.27 | 4.52% | 0.976 | 79.0% |
| Gradient Boosting 🥈 | 4,981.12 | 3,676.43 | 4.65% | 0.975 | 76.3% |
| Linear Regression 🥉 | 6,186.23 | 4,494.53 | 5.64% | 0.961 | 73.4% |
| LSTM | 7,446.65 | 5,535.47 | 7.38% | 0.944 | 55.9% |

## 🏗️ Project Structure

```
nc-traffic-ml/
├── 📁 dashboard/                 # Streamlit interactive dashboard
│   └── app.py                   # Main dashboard application
├── 📁 data/                     # Data storage
│   ├── 📁 models/              # Trained model files
│   ├── 📁 processed/           # Processed datasets
│   └── 📁 raw/                 # Raw data files
├── 📁 notebooks/               # Jupyter notebooks for exploration
├── 📁 src/                     # Core Python modules
│   ├── data_processing.py      # Data generation and preprocessing
│   ├── traditional_ml.py       # Traditional ML models
│   ├── deep_learning.py        # LSTM deep learning models
│   ├── evaluation.py           # Model evaluation and metrics
│   └── visualization.py        # Plotting and visualization
├── main.py                     # Main pipeline execution script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.10-3.13** (TensorFlow compatibility)
- **Git** for cloning the repository
- **8GB+ RAM** recommended for LSTM training

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/chigoziennani/nc-traffic-ml.git
   cd nc-traffic-ml
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install TensorFlow** (for Python 3.13)
   ```bash
   pip install tf-nightly
   ```

### Running the Pipeline

1. **Execute the complete pipeline**
   ```bash
   python main.py
   ```

2. **Launch the interactive dashboard**
   ```bash
   streamlit run dashboard/app.py
   ```

3. **Access the dashboard**
   Open your browser and go to: `http://localhost:8501`

## 📈 Usage Examples

### Basic Pipeline Execution

```python
from main import TrafficForecastingPipeline

# Initialize pipeline
pipeline = TrafficForecastingPipeline()

# Run complete pipeline
results = pipeline.run_complete_pipeline(
    start_year=2010,
    end_year=2023,
    forecast_days=30
)

# Print results
print(f"Best Model: {results['evaluation_results'].iloc[0]['Model']}")
print(f"RMSE: {results['evaluation_results'].iloc[0]['RMSE']:.2f}")
```

### Individual Component Usage

```python
# Data Processing
from src.data_processing import NCDataProcessor
processor = NCDataProcessor()
data = processor.generate_synthetic_data(2010, 2023)

# Traditional ML
from src.traditional_ml import TraditionalMLModels
ml_models = TraditionalMLModels()
features, target = ml_models.prepare_features(data)
results = ml_models.train_models(features, target)

# LSTM Deep Learning
from src.deep_learning import LSTMTrafficForecaster
lstm = LSTMTrafficForecaster()
X, y = lstm.prepare_sequences(data)
lstm_results = lstm.train_model(X, y)

# Evaluation
from src.evaluation import ModelEvaluator
evaluator = ModelEvaluator()
metrics = evaluator.calculate_metrics(y_true, y_pred, 'model_name')
```

## 🔧 Configuration

### Model Parameters

**Traditional ML Models:**
```python
# Random Forest
n_estimators=100
max_depth=15
min_samples_split=5

# Gradient Boosting
n_estimators=100
learning_rate=0.1
max_depth=6

# Linear Regression
# Uses default sklearn parameters
```

**LSTM Model:**
```python
# Architecture
lstm_units=[128, 64, 32]
dropout_rate=0.2
learning_rate=0.001

# Training
epochs=50
batch_size=32
validation_split=0.2
sequence_length=30
```

### Feature Engineering

The pipeline creates 29 engineered features:

- **Temporal Features**: Year, month, day, weekday, weekend, holiday
- **Cyclical Encoding**: Sin/cos transformations for periodic patterns
- **Lag Features**: 1, 7, 14, 30, 365-day lags
- **Rolling Statistics**: Mean, std, min, max over 7, 14, 30, 90-day windows

## 📊 Data Sources

### Highway Segments

The pipeline generates synthetic data for 5 major NC highway segments:

1. **I-40 Raleigh-Durham** - Major interstate corridor
2. **I-85 Charlotte** - High-traffic urban interstate
3. **I-95 Fayetteville** - East coast corridor
4. **NC-147 Durham** - State highway
5. **US-421 Winston-Salem** - US highway

### Data Characteristics

- **Time Period**: 2010-2023 (13 years)
- **Frequency**: Daily observations
- **Records**: ~23,740 total records
- **Features**: 29 engineered features
- **Target**: AADT (Average Annual Daily Traffic)

## 🎯 Model Performance

### Detailed Metrics

**Random Forest (Best Model):**
- **RMSE**: 4,895.83 vehicles/day
- **MAE**: 3,606.27 vehicles/day
- **MAPE**: 4.52%
- **R²**: 0.976 (97.6% accuracy)
- **Directional Accuracy**: 79.0%
- **Theil's U**: 0.058

### Model Comparison

The Random Forest model outperforms all others due to:
- **Non-linear pattern capture**: Handles complex traffic patterns
- **Feature importance**: Leverages engineered features effectively
- **Robustness**: Less sensitive to outliers than linear models
- **Interpretability**: Provides feature importance rankings

## 🛠️ Technical Implementation

### Architecture

```python
class TrafficForecastingPipeline:
    def __init__(self):
        self.processor = NCDataProcessor()
        self.traditional_models = TraditionalMLModels()
        self.lstm_model = LSTMTrafficForecaster()
        self.evaluator = ModelEvaluator()
        self.visualizer = TrafficVisualizer()
```

### Key Components

1. **Data Processing Module**
   - Synthetic data generation with realistic patterns
   - Feature engineering and preprocessing
   - Data validation and cleaning

2. **Traditional ML Module**
   - Scikit-learn pipeline with feature selection
   - Cross-validation and hyperparameter tuning
   - Model persistence and loading

3. **Deep Learning Module**
   - LSTM architecture with batch normalization
   - Early stopping and learning rate scheduling
   - Sequence preparation and scaling

4. **Evaluation Module**
   - Multiple performance metrics
   - Model comparison and ranking
   - Statistical significance testing

5. **Visualization Module**
   - Interactive plots and charts
   - Time series analysis
   - Model performance visualization

## 📱 Interactive Dashboard

The Streamlit dashboard provides:

- **Real-time Traffic Visualization**: Interactive time series plots
- **Model Performance Comparison**: Side-by-side model metrics
- **Forecast Exploration**: Future traffic predictions
- **Feature Analysis**: Importance rankings and correlations
- **Data Exploration**: Raw and processed data inspection

### Dashboard Features

- **Responsive Design**: Works on desktop and mobile
- **Real-time Updates**: Live data refresh capabilities
- **Interactive Charts**: Zoom, pan, and hover functionality
- **Export Options**: Download plots and data
- **Customizable Views**: Filter by date range and segments

## 🔍 Model Interpretability

### Feature Importance (Random Forest)

Top 10 most important features:
1. `aadt_lag_1` - Previous day's traffic
2. `aadt_rolling_mean_30` - 30-day average
3. `aadt_lag_7` - Weekly lag
4. `month_sin` - Seasonal pattern
5. `aadt_rolling_std_30` - 30-day volatility
6. `day_of_week_sin` - Weekly pattern
7. `aadt_lag_14` - Bi-weekly lag
8. `aadt_rolling_mean_7` - Weekly average
9. `is_weekend` - Weekend indicator
10. `aadt_lag_365` - Yearly lag

### Insights

- **Temporal Dependencies**: Recent history (1-7 days) is most predictive
- **Seasonal Patterns**: Monthly and weekly cycles significantly impact traffic
- **Volatility**: Rolling standard deviations capture traffic variability
- **Special Events**: Weekend and holiday effects are important

## 🚀 Deployment

### Production Setup

1. **Model Serving**
   ```python
   # Load trained models
   rf_model = joblib.load('data/models/random_forest.joblib')
   lstm_model = load_model('data/models/lstm_model.h5')
   
   # Make predictions
   prediction = rf_model.predict(features)
   ```

2. **API Development**
   ```python
   from flask import Flask, request, jsonify
   
   app = Flask(__name__)
   
   @app.route('/predict', methods=['POST'])
   def predict():
       data = request.json
       features = preprocess_features(data)
       prediction = model.predict(features)
       return jsonify({'prediction': prediction.tolist()})
   ```

3. **Docker Containerization**
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 8501
   CMD ["streamlit", "run", "dashboard/app.py"]
   ```

### Monitoring

- **Model Drift Detection**: Monitor prediction accuracy over time
- **Data Quality Checks**: Validate incoming data
- **Performance Metrics**: Track RMSE, MAE, and other KPIs
- **Alert Systems**: Notify when models need retraining

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Commit your changes**: `git commit -m 'Add amazing feature'`
5. **Push to the branch**: `git push origin feature/amazing-feature`
6. **Open a Pull Request**

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 src/
black src/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NCDOT**: For traffic data patterns and insights
- **TensorFlow Team**: For LSTM implementation support
- **Scikit-learn Community**: For traditional ML algorithms
- **Streamlit Team**: For the interactive dashboard framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/nc-traffic-ml/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/nc-traffic-ml/discussions)
- **Email**: your.email@example.com

## 📚 References

1. **Traffic Forecasting Literature**
   - [Traffic Flow Prediction with LSTM](https://doi.org/10.1016/j.trc.2017.09.017)
   - [Machine Learning for Transportation](https://doi.org/10.1016/j.trc.2018.12.015)

2. **Technical Documentation**
   - [TensorFlow LSTM Guide](https://www.tensorflow.org/guide/keras/rnn)
   - [Scikit-learn Pipeline](https://scikit-learn.org/stable/modules/compose.html)
   - [Streamlit Documentation](https://docs.streamlit.io/)

3. **Data Sources**
   - [NCDOT Traffic Data](https://www.ncdot.gov/divisions/planning/traffic-data/)
   - [FHWA Traffic Monitoring](https://www.fhwa.dot.gov/policyinformation/travel_monitoring/)

---

**Made with ❤️ for North Carolina Traffic Analysis**

*Last updated: June 2025* 