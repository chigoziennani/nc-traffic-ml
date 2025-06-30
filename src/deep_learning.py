"""
Deep Learning Models for NC Traffic Forecasting

This module implements LSTM neural networks using TensorFlow for
advanced traffic volume prediction with temporal dependencies.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import logging
from typing import Dict, List, Tuple, Any, Optional
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class LSTMTrafficForecaster:
    """LSTM-based traffic forecasting model."""
    
    def __init__(self, models_dir: str = "data/models"):
        """
        Initialize the LSTM forecaster.
        
        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.sequence_length = 30  # Number of time steps to look back
        self.results = {}
        
    def prepare_sequences(self, df: pd.DataFrame, target_col: str = 'aadt', 
                         sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            sequence_length: Number of time steps in each sequence
            
        Returns:
            Tuple of (X, y) arrays for LSTM training
        """
        logger.info(f"Preparing LSTM sequences with length {sequence_length}...")
        
        self.sequence_length = sequence_length
        
        # Create a copy to avoid modifying original data
        data = df.copy()
        
        # Sort by segment and date
        data = data.sort_values(['segment_id', 'date']).reset_index(drop=True)
        
        # Prepare features (exclude non-numeric columns)
        feature_cols = [col for col in data.columns if col not in [
            'date', 'segment_id', 'segment_name', target_col
        ] and data[col].dtype in ['int64', 'float64']]
        
        # Scale features
        features_scaled = self.feature_scaler.fit_transform(data[feature_cols])
        
        # Scale target
        target_scaled = self.scaler.fit_transform(data[[target_col]])
        
        X, y = [], []
        
        # Create sequences for each segment
        for segment_id in data['segment_id'].unique():
            segment_data = data[data['segment_id'] == segment_id]
            segment_features = features_scaled[data['segment_id'] == segment_id]
            segment_target = target_scaled[data['segment_id'] == segment_id]
            
            # Create sequences
            for i in range(sequence_length, len(segment_data)):
                X.append(segment_features[i-sequence_length:i])
                y.append(segment_target[i])
        
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        logger.info(f"Created {len(X)} sequences with shape {X.shape}")
        
        return X, y
    
    def build_lstm_model(self, input_shape: Tuple[int, int], 
                        lstm_units: List[int] = [128, 64, 32],
                        dropout_rate: float = 0.2,
                        learning_rate: float = 0.001) -> Sequential:
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: Shape of input sequences (timesteps, features)
            lstm_units: List of LSTM units for each layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            
        Returns:
            Compiled LSTM model
        """
        logger.info(f"Building LSTM model with input shape {input_shape}...")
        
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=lstm_units[0],
            return_sequences=True if len(lstm_units) > 1 else False,
            input_shape=input_shape,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate
        ))
        model.add(BatchNormalization())
        
        # Additional LSTM layers
        for i, units in enumerate(lstm_units[1:], 1):
            model.add(LSTM(
                units=units,
                return_sequences=i < len(lstm_units) - 1,
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate
            ))
            model.add(BatchNormalization())
        
        # Dense layers
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='linear'))
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        logger.info("LSTM model built successfully")
        model.summary()
        
        return model
    
    def train_model(self, X: np.ndarray, y: np.ndarray,
                   validation_split: float = 0.2,
                   epochs: int = 100,
                   batch_size: int = 32,
                   patience: int = 15) -> Dict[str, Any]:
        """
        Train the LSTM model.
        
        Args:
            X: Input sequences
            y: Target values
            validation_split: Proportion of data for validation
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            patience: Early stopping patience
            
        Returns:
            Training history and results
        """
        logger.info("Training LSTM model...")
        
        # Build model
        self.model = self.build_lstm_model(
            input_shape=(X.shape[1], X.shape[2])
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience//2,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.models_dir, 'lstm_best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Make predictions on training data
        train_predictions = self.model.predict(X)
        
        # Inverse transform predictions
        train_predictions_original = self.scaler.inverse_transform(train_predictions)
        y_original = self.scaler.inverse_transform(y)
        
        # Calculate metrics
        mse = mean_squared_error(y_original, train_predictions_original)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_original, train_predictions_original)
        mape = np.mean(np.abs((y_original - train_predictions_original) / y_original)) * 100
        r2 = r2_score(y_original, train_predictions_original)
        
        self.results = {
            'history': history.history,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'predictions': train_predictions_original.flatten(),
            'actual': y_original.flatten()
        }
        
        logger.info(f"LSTM Training Results - RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%, R²: {r2:.3f}")
        
        return self.results
    
    def predict_future(self, df: pd.DataFrame, forecast_days: int = 30) -> pd.DataFrame:
        """
        Make future predictions using the trained LSTM model.
        
        Args:
            df: Historical data DataFrame
            forecast_days: Number of days to forecast
            
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        logger.info(f"Making LSTM future predictions for {forecast_days} days...")
        
        # Get the last date in the data
        last_date = df['date'].max()
        
        # Create future dates
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
        # Prepare features for the last sequence
        data = df.copy()
        data = data.sort_values(['segment_id', 'date']).reset_index(drop=True)
        
        # Get feature columns
        feature_cols = [col for col in data.columns if col not in [
            'date', 'segment_id', 'segment_name', 'aadt'
        ] and data[col].dtype in ['int64', 'float64']]
        
        predictions_list = []
        
        # Make predictions for each segment
        for segment_id in data['segment_id'].unique():
            segment_data = data[data['segment_id'] == segment_id]
            segment_name = segment_data['segment_name'].iloc[0]
            
            # Get the last sequence for this segment
            if len(segment_data) >= self.sequence_length:
                last_sequence = segment_data[feature_cols].iloc[-self.sequence_length:].values
                last_sequence_scaled = self.feature_scaler.transform(last_sequence)
                
                # Make predictions for future days
                current_sequence = last_sequence_scaled.copy()
                
                for i, future_date in enumerate(future_dates):
                    # Reshape sequence for prediction
                    sequence_input = current_sequence.reshape(1, self.sequence_length, len(feature_cols))
                    
                    # Make prediction
                    prediction_scaled = self.model.predict(sequence_input, verbose=0)
                    prediction = self.scaler.inverse_transform(prediction_scaled)[0][0]
                    
                    # Add prediction to results
                    predictions_list.append({
                        'date': future_date,
                        'segment_id': segment_id,
                        'segment_name': segment_name,
                        'lstm_prediction': prediction
                    })
                    
                    # Update sequence for next prediction (simplified approach)
                    # In practice, you'd need to update all features properly
                    if i < len(future_dates) - 1:
                        # Create a simple update for the sequence
                        new_row = current_sequence[-1].copy()
                        # Update time-based features (simplified)
                        new_row[0] = (future_date.year - 2010)  # year feature
                        new_row[1] = future_date.month  # month feature
                        new_row[2] = future_date.dayofyear  # day_of_year feature
                        new_row[3] = future_date.weekday()  # day_of_week feature
                        
                        # Update sequence
                        current_sequence = np.vstack([current_sequence[1:], new_row.reshape(1, -1)])
        
        predictions_df = pd.DataFrame(predictions_list)
        logger.info(f"Generated LSTM predictions for {len(predictions_df)} future data points")
        
        return predictions_df
    
    def save_model(self) -> None:
        """Save the trained LSTM model and scalers."""
        logger.info("Saving LSTM model and scalers...")
        
        # Save model
        model_path = os.path.join(self.models_dir, "lstm_model.h5")
        self.model.save(model_path)
        logger.info(f"Saved LSTM model to {model_path}")
        
        # Save scalers
        scaler_path = os.path.join(self.models_dir, "lstm_scalers.joblib")
        joblib.dump({
            'target_scaler': self.scaler,
            'feature_scaler': self.feature_scaler,
            'sequence_length': self.sequence_length
        }, scaler_path)
        logger.info(f"Saved LSTM scalers to {scaler_path}")
    
    def load_model(self) -> None:
        """Load the trained LSTM model and scalers."""
        logger.info("Loading LSTM model and scalers...")
        
        # Load model
        model_path = os.path.join(self.models_dir, "lstm_model.h5")
        if os.path.exists(model_path):
            self.model = load_model(model_path)
            logger.info(f"Loaded LSTM model from {model_path}")
        
        # Load scalers
        scaler_path = os.path.join(self.models_dir, "lstm_scalers.joblib")
        if os.path.exists(scaler_path):
            scalers = joblib.load(scaler_path)
            self.scaler = scalers['target_scaler']
            self.feature_scaler = scalers['feature_scaler']
            self.sequence_length = scalers['sequence_length']
            logger.info(f"Loaded LSTM scalers from {scaler_path}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of LSTM model performance.
        
        Returns:
            Dictionary with model performance summary
        """
        if not self.results:
            return {}
        
        return {
            'RMSE': self.results['rmse'],
            'MAE': self.results['mae'],
            'MAPE': self.results['mape'],
            'R²': self.results['r2'],
            'Final_Loss': self.results['history']['loss'][-1] if 'history' in self.results else None,
            'Final_Val_Loss': self.results['history']['val_loss'][-1] if 'history' in self.results else None
        }


class EnsembleForecaster:
    """Ensemble model combining traditional ML and LSTM predictions."""
    
    def __init__(self, traditional_models, lstm_model):
        """
        Initialize ensemble forecaster.
        
        Args:
            traditional_models: Trained traditional ML models
            lstm_model: Trained LSTM model
        """
        self.traditional_models = traditional_models
        self.lstm_model = lstm_model
        
    def ensemble_predict(self, df: pd.DataFrame, forecast_days: int = 30,
                        weights: Dict[str, float] = None) -> pd.DataFrame:
        """
        Make ensemble predictions.
        
        Args:
            df: Historical data DataFrame
            forecast_days: Number of days to forecast
            weights: Weights for each model (optional)
            
        Returns:
            DataFrame with ensemble predictions
        """
        logger.info("Making ensemble predictions...")
        
        # Get predictions from traditional models
        traditional_predictions = self.traditional_models.predict_future(df, forecast_days)
        
        # Get predictions from LSTM
        lstm_predictions = self.lstm_model.predict_future(df, forecast_days)
        
        # Merge predictions
        ensemble_df = traditional_predictions.merge(
            lstm_predictions[['date', 'segment_id', 'lstm_prediction']],
            on=['date', 'segment_id'],
            how='inner'
        )
        
        # Calculate ensemble prediction (simple average)
        prediction_cols = [col for col in ensemble_df.columns if 'prediction' in col]
        
        if weights is None:
            # Equal weights
            weights = {col: 1.0/len(prediction_cols) for col in prediction_cols}
        
        # Calculate weighted ensemble
        ensemble_df['ensemble_prediction'] = 0
        for col in prediction_cols:
            ensemble_df['ensemble_prediction'] += ensemble_df[col] * weights.get(col, 1.0/len(prediction_cols))
        
        logger.info(f"Generated ensemble predictions for {len(ensemble_df)} future data points")
        
        return ensemble_df


def main():
    """Main function to demonstrate LSTM training."""
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
    
    # Initialize LSTM model
    lstm_model = LSTMTrafficForecaster()
    
    # Prepare sequences
    X, y = lstm_model.prepare_sequences(data, sequence_length=30)
    
    # Train model
    results = lstm_model.train_model(X, y, epochs=50, patience=10)
    
    # Save model
    lstm_model.save_model()
    
    # Get model summary
    summary = lstm_model.get_model_summary()
    logger.info("LSTM Model Performance Summary:")
    for metric, value in summary.items():
        if value is not None:
            logger.info(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")


if __name__ == "__main__":
    main() 