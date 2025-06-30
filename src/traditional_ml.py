"""
Traditional Machine Learning Models for NC Traffic Forecasting

This module implements classical machine learning approaches including
Linear Regression, Random Forest, and Gradient Boosting for traffic volume prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
import os
import logging
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TraditionalMLModels:
    """Traditional ML models for traffic forecasting."""
    
    def __init__(self, models_dir: str = "data/models"):
        """
        Initialize the traditional ML models.
        
        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = None
        self.results = {}
        
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'aadt') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for modeling.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            
        Returns:
            Tuple of features DataFrame and target Series
        """
        logger.info("Preparing features for traditional ML models...")
        
        # Create a copy to avoid modifying original data
        data = df.copy()
        
        # Separate features and target
        target = data[target_col]
        
        # Remove target and non-feature columns
        feature_cols = [col for col in data.columns if col not in [
            target_col, 'date', 'segment_id', 'segment_name'
        ]]
        
        features = data[feature_cols]
        
        # Handle categorical variables
        categorical_cols = features.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                features[col] = self.label_encoders[col].fit_transform(features[col])
            else:
                features[col] = self.label_encoders[col].transform(features[col])
        
        # Handle NaN values in features
        for col in features.columns:
            if features[col].isna().any():
                if 'lag' in col:
                    # For lag features, forward fill from the last known value
                    features[col] = features[col].fillna(method='ffill')
                elif 'rolling' in col:
                    # For rolling features, fill with the mean of the column
                    features[col] = features[col].fillna(features[col].mean())
                else:
                    # For other features, fill with 0
                    features[col] = features[col].fillna(0)
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        # Final NaN handling - fill any remaining NaNs with 0
        features = features.fillna(0)
        
        logger.info(f"Prepared {len(features.columns)} features for modeling")
        logger.info(f"Feature names: {self.feature_names}")
        
        return features, target
    
    def create_pipelines(self) -> Dict[str, Pipeline]:
        """
        Create scikit-learn pipelines for different models.
        
        Returns:
            Dictionary of model pipelines
        """
        pipelines = {}
        
        # Linear Regression Pipeline
        linear_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(score_func=f_regression, k=20)),
            ('regressor', LinearRegression())
        ])
        pipelines['linear_regression'] = linear_pipeline
        
        # Random Forest Pipeline
        rf_pipeline = Pipeline([
            ('regressor', RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ))
        ])
        pipelines['random_forest'] = rf_pipeline
        
        # Gradient Boosting Pipeline
        gb_pipeline = Pipeline([
            ('regressor', GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ))
        ])
        pipelines['gradient_boosting'] = gb_pipeline
        
        return pipelines
    
    def train_models(self, features: pd.DataFrame, target: pd.Series, 
                    test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train traditional ML models.
        
        Args:
            features: Feature DataFrame
            target: Target Series
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with training results
        """
        logger.info("Training traditional ML models...")
        
        # Create pipelines
        pipelines = self.create_pipelines()
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        results = {}
        
        for model_name, pipeline in pipelines.items():
            logger.info(f"Training {model_name}...")
            
            # Store pipeline
            self.models[model_name] = pipeline
            
            # Cross-validation scores
            cv_scores = cross_val_score(
                pipeline, features, target, 
                cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1
            )
            cv_rmse = np.sqrt(-cv_scores)
            
            # Fit the model on full training data
            pipeline.fit(features, target)
            
            # Make predictions
            predictions = pipeline.predict(features)
            
            # Calculate metrics
            mse = mean_squared_error(target, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(target, predictions)
            mape = np.mean(np.abs((target - predictions) / target)) * 100
            r2 = r2_score(target, predictions)
            
            results[model_name] = {
                'cv_rmse_mean': cv_rmse.mean(),
                'cv_rmse_std': cv_rmse.std(),
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'r2': r2,
                'predictions': predictions
            }
            
            logger.info(f"{model_name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%, R²: {r2:.3f}")
        
        self.results = results
        return results
    
    def predict_future(self, df: pd.DataFrame, forecast_days: int = 30) -> pd.DataFrame:
        """
        Make future predictions using trained models.
        
        Args:
            df: Historical data DataFrame
            forecast_days: Number of days to forecast
            
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Making future predictions for {forecast_days} days...")
        
        # Get the last date in the data
        last_date = df['date'].max()
        
        # Create future dates
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
        # Create future data frame
        future_data = []
        for date in future_dates:
            for segment_id in df['segment_id'].unique():
                segment_data = df[df['segment_id'] == segment_id].iloc[-1].copy()
                segment_data['date'] = date
                segment_data['aadt'] = np.nan  # Will be predicted
                future_data.append(segment_data)
        
        future_df = pd.DataFrame(future_data)
        
        # Combine historical and future data
        combined_df = pd.concat([df, future_df], ignore_index=True)
        
        # Sort by segment and date
        combined_df = combined_df.sort_values(['segment_id', 'date']).reset_index(drop=True)
        
        # Recalculate features for future data
        # This is a simplified approach - in practice, you'd need more sophisticated feature engineering
        for segment_id in combined_df['segment_id'].unique():
            segment_mask = combined_df['segment_id'] == segment_id
            segment_data = combined_df[segment_mask].copy()
            
            # Update time features
            segment_data['year'] = segment_data['date'].dt.year
            segment_data['month'] = segment_data['date'].dt.month
            segment_data['day_of_year'] = segment_data['date'].dt.dayofyear
            segment_data['day_of_week'] = segment_data['date'].dt.weekday
            segment_data['quarter'] = segment_data['date'].dt.quarter
            segment_data['week_of_year'] = segment_data['date'].dt.isocalendar().week
            
            # Cyclical encoding
            segment_data['month_sin'] = np.sin(2 * np.pi * segment_data['month'] / 12)
            segment_data['month_cos'] = np.cos(2 * np.pi * segment_data['month'] / 12)
            segment_data['day_of_year_sin'] = np.sin(2 * np.pi * segment_data['day_of_year'] / 365.25)
            segment_data['day_of_year_cos'] = np.cos(2 * np.pi * segment_data['day_of_year'] / 365.25)
            segment_data['day_of_week_sin'] = np.sin(2 * np.pi * segment_data['day_of_week'] / 7)
            segment_data['day_of_week_cos'] = np.cos(2 * np.pi * segment_data['day_of_week'] / 7)
            
            # Update lag features (use historical data)
            for lag in [1, 7, 14, 30]:
                segment_data[f'aadt_lag_{lag}'] = segment_data['aadt'].shift(lag)
            
            segment_data['aadt_lag_365'] = segment_data['aadt'].shift(365)
            
            # Update rolling features
            for window in [7, 14, 30, 90]:
                segment_data[f'aadt_rolling_mean_{window}'] = segment_data['aadt'].rolling(window=window, min_periods=1).mean()
                segment_data[f'aadt_rolling_std_{window}'] = segment_data['aadt'].rolling(window=window, min_periods=1).std()
            
            segment_data['aadt_rolling_min_30'] = segment_data['aadt'].rolling(window=30, min_periods=1).min()
            segment_data['aadt_rolling_max_30'] = segment_data['aadt'].rolling(window=30, min_periods=1).max()
            
            # Update combined dataframe
            combined_df.loc[segment_mask] = segment_data
        
        # Prepare features for prediction
        future_features, _ = self.prepare_features(combined_df)
        
        # Handle NaN values in future features
        # For lag features, fill with the last known value
        # For rolling features, fill with the mean of the last few values
        for col in future_features.columns:
            if future_features[col].isna().any():
                if 'lag' in col:
                    # For lag features, forward fill from the last known value
                    future_features[col] = future_features[col].fillna(method='ffill')
                elif 'rolling' in col:
                    # For rolling features, fill with the mean of the last 30 values
                    last_values = combined_df[col.replace('_prediction', '')].dropna().tail(30)
                    if len(last_values) > 0:
                        future_features[col] = future_features[col].fillna(last_values.mean())
                    else:
                        # If no historical data, fill with 0
                        future_features[col] = future_features[col].fillna(0)
                else:
                    # For other features, fill with 0
                    future_features[col] = future_features[col].fillna(0)
        
        # Make predictions for each model
        predictions_df = combined_df[['date', 'segment_id', 'segment_name']].copy()
        
        for model_name, model in self.models.items():
            if model_name in self.results:
                pred = model.predict(future_features)
                predictions_df[f'{model_name}_prediction'] = pred
        
        # Filter to only future dates
        future_predictions = predictions_df[predictions_df['date'] > last_date].copy()
        
        logger.info(f"Generated predictions for {len(future_predictions)} future data points")
        
        return future_predictions
    
    def save_models(self) -> None:
        """Save trained models to disk."""
        logger.info("Saving trained models...")
        
        for model_name, model in self.models.items():
            model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} to {model_path}")
        
        # Save scalers and encoders
        scaler_path = os.path.join(self.models_dir, "scalers.joblib")
        joblib.dump(self.scalers, scaler_path)
        
        encoder_path = os.path.join(self.models_dir, "label_encoders.joblib")
        joblib.dump(self.label_encoders, encoder_path)
        
        # Save feature names
        feature_path = os.path.join(self.models_dir, "feature_names.txt")
        with open(feature_path, 'w') as f:
            for feature in self.feature_names:
                f.write(f"{feature}\n")
    
    def load_models(self) -> None:
        """Load trained models from disk."""
        logger.info("Loading trained models...")
        
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.joblib') and not f.startswith('scalers') and not f.startswith('label_encoders')]
        
        for model_file in model_files:
            model_name = model_file.replace('.joblib', '')
            model_path = os.path.join(self.models_dir, model_file)
            self.models[model_name] = joblib.load(model_path)
            logger.info(f"Loaded {model_name} from {model_path}")
        
        # Load scalers and encoders
        scaler_path = os.path.join(self.models_dir, "scalers.joblib")
        if os.path.exists(scaler_path):
            self.scalers = joblib.load(scaler_path)
        
        encoder_path = os.path.join(self.models_dir, "label_encoders.joblib")
        if os.path.exists(encoder_path):
            self.label_encoders = joblib.load(encoder_path)
        
        # Load feature names
        feature_path = os.path.join(self.models_dir, "feature_names.txt")
        if os.path.exists(feature_path):
            with open(feature_path, 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
    
    def get_feature_importance(self, model_name: str = 'random_forest') -> pd.DataFrame:
        """
        Get feature importance for tree-based models.
        
        Args:
            model_name: Name of the model to analyze
            
        Returns:
            DataFrame with feature importance scores
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if hasattr(model, 'named_steps'):
            # Pipeline
            regressor = model.named_steps['regressor']
        else:
            # Direct model
            regressor = model
        
        if hasattr(regressor, 'feature_importances_'):
            importance = regressor.feature_importances_
            feature_names = self.feature_names
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            logger.warning(f"Model {model_name} does not support feature importance")
            return pd.DataFrame()
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of all model performances.
        
        Returns:
            Dictionary with model performance summary
        """
        summary = {}
        
        for model_name, results in self.results.items():
            summary[model_name] = {
                'RMSE': results['rmse'],
                'MAE': results['mae'],
                'MAPE': results['mape'],
                'R²': results['r2'],
                'CV_RMSE_Mean': results['cv_rmse_mean'],
                'CV_RMSE_Std': results['cv_rmse_std']
            }
        
        return summary


def main():
    """Main function to demonstrate traditional ML training."""
    from data_processing import NCDataProcessor
    
    # Load data
    processor = NCDataProcessor()
    try:
        data = processor.load_data("nc_traffic_processed.csv", "processed")
    except FileNotFoundError:
        logger.info("Processed data not found. Running data processing first...")
        from data_processing import main as process_data
        process_data()
        data = processor.load_data("nc_traffic_processed.csv", "processed")
    
    # Initialize traditional ML models
    ml_models = TraditionalMLModels()
    
    # Prepare features
    features, target = ml_models.prepare_features(data)
    
    # Train models
    results = ml_models.train_models(features, target)
    
    # Save models
    ml_models.save_models()
    
    # Get model summary
    summary = ml_models.get_model_summary()
    logger.info("Model Performance Summary:")
    for model_name, metrics in summary.items():
        logger.info(f"{model_name}: RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, MAPE={metrics['MAPE']:.2f}%, R²={metrics['R²']:.3f}")
    
    # Get feature importance for Random Forest
    importance_df = ml_models.get_feature_importance('random_forest')
    if not importance_df.empty:
        logger.info("Top 10 Most Important Features (Random Forest):")
        logger.info(importance_df.head(10))


if __name__ == "__main__":
    main() 