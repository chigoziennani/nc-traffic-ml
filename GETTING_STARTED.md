# Getting Started with NC Traffic Forecasting

## Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/nc-traffic-ml.git
   cd nc-traffic-ml
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install tf-nightly  # For Python 3.13
   ```

3. **Run the pipeline**
   ```bash
   python main.py
   ```

4. **Launch dashboard**
   ```bash
   streamlit run dashboard/app.py
   ```

## What's Included

- Complete ML pipeline (data processing, training, evaluation)
- 4 trained models (Linear Regression, Random Forest, Gradient Boosting, LSTM)
- Interactive Streamlit dashboard
- Comprehensive documentation
- Sample data and results

## Model Performance

| Model | RMSE | R² | Status |
|-------|------|----|--------|
| Random Forest | 4,895.83 | 0.976 | Best |
| Gradient Boosting | 4,981.12 | 0.975 | Runner-up |
| Linear Regression | 6,186.23 | 0.961 | Good |
| LSTM | 7,446.65 | 0.944 | Experimental |

## Support

- Read the full README.md for detailed documentation
- Report issues on GitHub
- Ask questions in Discussions

Happy forecasting!
