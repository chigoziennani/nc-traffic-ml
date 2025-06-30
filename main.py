"""
Main Execution Script for NC Traffic Forecasting

This script orchestrates the entire traffic forecasting pipeline including
data processing, model training, evaluation, and visualization.
"""

import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import NCDataProcessor
from traditional_ml import TraditionalMLModels
from deep_learning import LSTMTrafficForecaster, EnsembleForecaster
from evaluation import ModelEvaluator
from visualization import TrafficVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('traffic_forecasting.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TrafficForecastingPipeline:
    """Complete traffic forecasting pipeline."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the pipeline.
        
        Args:
            data_dir: Directory for data storage
        """
        self.data_dir = data_dir
        self.processor = NCDataProcessor(data_dir)
        self.traditional_models = TraditionalMLModels()
        self.lstm_model = LSTMTrafficForecaster()
        self.evaluator = ModelEvaluator()
        self.visualizer = TrafficVisualizer()
        
        # Results storage
        self.data = None
        self.training_results = {}
        self.forecast_results = {}
        
    def run_data_processing(self, start_year: int = 2010, end_year: int = 2023) -> pd.DataFrame:
        """
        Run the data processing pipeline.
        
        Args:
            start_year: Start year for data generation
            end_year: End year for data generation
            
        Returns:
            Processed DataFrame
        """
        logger.info("=" * 60)
        logger.info("STEP 1: DATA PROCESSING")
        logger.info("=" * 60)
        
        try:
            # Try to load existing processed data
            self.data = self.processor.load_data("nc_traffic_processed.csv", "processed")
            logger.info("Loaded existing processed data")
        except FileNotFoundError:
            logger.info("No existing processed data found. Generating new data...")
            
            # Generate synthetic data
            raw_data = self.processor.generate_synthetic_traffic_data(start_year, end_year)
            self.processor.save_data(raw_data, "nc_traffic_raw.csv", "raw")
            
            # Preprocess data
            self.data = self.processor.preprocess_data(raw_data)
            self.processor.save_data(self.data, "nc_traffic_processed.csv", "processed")
        
        # Display data summary
        summary = self.processor.get_data_summary(self.data)
        logger.info("Data Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        
        return self.data
    
    def run_traditional_ml_training(self) -> dict:
        """
        Run traditional ML model training.
        
        Returns:
            Training results dictionary
        """
        logger.info("=" * 60)
        logger.info("STEP 2: TRADITIONAL ML TRAINING")
        logger.info("=" * 60)
        
        # Prepare features
        features, target = self.traditional_models.prepare_features(self.data)
        
        # Train models
        results = self.traditional_models.train_models(features, target)
        
        # Save models
        self.traditional_models.save_models()
        
        # Store results
        self.training_results.update(results)
        
        return results
    
    def run_lstm_training(self, sequence_length: int = 30, epochs: int = 50) -> dict:
        """
        Run LSTM model training.
        
        Args:
            sequence_length: Length of input sequences
            epochs: Number of training epochs
            
        Returns:
            Training results dictionary
        """
        logger.info("=" * 60)
        logger.info("STEP 3: LSTM DEEP LEARNING TRAINING")
        logger.info("=" * 60)
        
        # Prepare sequences
        X, y = self.lstm_model.prepare_sequences(self.data, sequence_length=sequence_length)
        
        # Train model
        results = self.lstm_model.train_model(X, y, epochs=epochs, patience=10)
        
        # Save model
        self.lstm_model.save_model()
        
        # Store results
        self.training_results['lstm'] = results
        
        return results
    
    def run_model_evaluation(self) -> pd.DataFrame:
        """
        Run comprehensive model evaluation.
        
        Returns:
            Model comparison DataFrame
        """
        logger.info("=" * 60)
        logger.info("STEP 4: MODEL EVALUATION")
        logger.info("=" * 60)
        
        # Prepare features for evaluation
        features, target = self.traditional_models.prepare_features(self.data)
        
        # Evaluate traditional models
        for model_name, model in self.traditional_models.models.items():
            predictions = model.predict(features)
            self.evaluator.calculate_metrics(target.values, predictions, model_name)
        
        # Evaluate LSTM model
        if 'lstm' in self.training_results:
            X, y = self.lstm_model.prepare_sequences(self.data)
            lstm_predictions = self.lstm_model.model.predict(X)
            lstm_predictions_original = self.lstm_model.scaler.inverse_transform(lstm_predictions)
            y_original = self.lstm_model.scaler.inverse_transform(y)
            
            # Ensure both arrays are 1D and have the same length
            lstm_predictions_flat = lstm_predictions_original.flatten()
            y_original_flat = y_original.flatten()
            
            # Use the minimum length to avoid shape mismatches
            min_length = min(len(lstm_predictions_flat), len(y_original_flat))
            
            if min_length > 0:
                # Use only the last min_length elements to ensure alignment
                aligned_predictions = lstm_predictions_flat[-min_length:]
                aligned_targets = y_original_flat[-min_length:]
                
                # Ensure no NaN values
                valid_mask = ~(np.isnan(aligned_predictions) | np.isnan(aligned_targets))
                if np.sum(valid_mask) > 0:
                    self.evaluator.calculate_metrics(
                        aligned_targets[valid_mask], 
                        aligned_predictions[valid_mask], 
                        'lstm'
                    )
                else:
                    logger.warning("No valid LSTM predictions for evaluation")
            else:
                logger.warning("LSTM predictions have zero length, skipping evaluation")
        
        # Compare models
        comparison = self.evaluator.compare_models(self.evaluator.results)
        
        # Generate evaluation report
        report = self.evaluator.generate_evaluation_report("evaluation_report.txt")
        logger.info("Evaluation report generated: evaluation_report.txt")
        
        return comparison
    
    def run_forecasting(self, forecast_days: int = 30) -> pd.DataFrame:
        """
        Run forecasting for future traffic volumes.
        
        Args:
            forecast_days: Number of days to forecast
            
        Returns:
            Forecast DataFrame
        """
        logger.info("=" * 60)
        logger.info("STEP 5: FUTURE FORECASTING")
        logger.info("=" * 60)
        
        # Traditional ML forecasts
        traditional_forecasts = self.traditional_models.predict_future(self.data, forecast_days)
        
        # LSTM forecasts
        lstm_forecasts = self.lstm_model.predict_future(self.data, forecast_days)
        
        # Merge forecasts
        self.forecast_results = traditional_forecasts.merge(
            lstm_forecasts[['date', 'segment_id', 'lstm_prediction']],
            on=['date', 'segment_id'],
            how='inner'
        )
        
        # Create ensemble forecast
        ensemble = EnsembleForecaster(self.traditional_models, self.lstm_model)
        ensemble_forecasts = ensemble.ensemble_predict(self.data, forecast_days)
        
        # Save forecasts
        self.forecast_results.to_csv("forecasts.csv", index=False)
        logger.info(f"Forecasts saved to forecasts.csv")
        
        return self.forecast_results
    
    def run_visualization(self) -> None:
        """
        Run comprehensive visualization pipeline.
        """
        logger.info("=" * 60)
        logger.info("STEP 6: VISUALIZATION")
        logger.info("=" * 60)
        
        # Create traffic trends plots
        self.visualizer.plot_traffic_trends(self.data, save_path="traffic_trends.png")
        
        # Create model comparison plots
        if self.evaluator.results:
            self.visualizer.plot_model_comparison(self.evaluator.results, save_path="model_comparison.png")
        
        # Create forecast plots
        if self.forecast_results is not None:
            self.visualizer.plot_forecasts(self.data, self.forecast_results, save_path="forecasts.png")
        
        # Create feature importance plot (for Random Forest)
        if hasattr(self.traditional_models, 'get_feature_importance'):
            importance_df = self.traditional_models.get_feature_importance('random_forest')
            if not importance_df.empty:
                self.visualizer.plot_feature_importance(importance_df, save_path="feature_importance.png")
        
        # Create correlation matrix
        self.visualizer.plot_correlation_matrix(self.data, save_path="correlation_matrix.png")
        
        logger.info("All visualizations completed and saved!")
    
    def run_complete_pipeline(self, start_year: int = 2010, end_year: int = 2023,
                             forecast_days: int = 30) -> dict:
        """
        Run the complete traffic forecasting pipeline.
        
        Args:
            start_year: Start year for data generation
            end_year: End year for data generation
            forecast_days: Number of days to forecast
            
        Returns:
            Pipeline results dictionary
        """
        logger.info("🚦 STARTING NC TRAFFIC FORECASTING PIPELINE 🚦")
        logger.info(f"Pipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Step 1: Data Processing
            self.run_data_processing(start_year, end_year)
            
            # Step 2: Traditional ML Training
            traditional_results = self.run_traditional_ml_training()
            
            # Step 3: LSTM Training
            lstm_results = self.run_lstm_training()
            
            # Step 4: Model Evaluation
            evaluation_results = self.run_model_evaluation()
            
            # Step 5: Forecasting
            forecast_results = self.run_forecasting(forecast_days)
            
            # Step 6: Visualization
            self.run_visualization()
            
            # Compile results
            pipeline_results = {
                'data_summary': self.processor.get_data_summary(self.data),
                'traditional_ml_results': traditional_results,
                'lstm_results': lstm_results,
                'evaluation_results': evaluation_results,
                'forecast_results': forecast_results,
                'pipeline_completed': True,
                'completion_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY! 🎉")
            logger.info(f"Pipeline completed at: {pipeline_results['completion_time']}")
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            raise


def main():
    """Main function to run the traffic forecasting pipeline."""
    
    # Initialize pipeline
    pipeline = TrafficForecastingPipeline()
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(
        start_year=2010,
        end_year=2023,
        forecast_days=30
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Data Records: {results['data_summary']['total_records']:,}")
    print(f"Highway Segments: {results['data_summary']['segments']}")
    print(f"Date Range: {results['data_summary']['date_range']['start']} to {results['data_summary']['date_range']['end']}")
    print(f"Models Trained: {len(results['traditional_ml_results'])} traditional + 1 LSTM")
    print(f"Forecast Days: 30")
    print(f"Pipeline Status: {'✅ COMPLETED' if results['pipeline_completed'] else '❌ FAILED'}")
    print("=" * 60)
    
    # Print best model
    if 'evaluation_results' in results and not results['evaluation_results'].empty:
        best_model = results['evaluation_results'].iloc[0]['Model']
        best_rmse = results['evaluation_results'].iloc[0]['RMSE']
        print(f"🏆 Best Model: {best_model} (RMSE: {best_rmse:.2f})")
    
    print("\n📁 Generated Files:")
    print("  • nc_traffic_raw.csv - Raw traffic data")
    print("  • nc_traffic_processed.csv - Processed data with features")
    print("  • forecasts.csv - Future traffic predictions")
    print("  • evaluation_report.txt - Model performance report")
    print("  • traffic_forecasting.log - Pipeline execution log")
    print("  • *.png - Visualization plots")
    print("  • data/models/ - Trained model files")
    
    print("\n🚀 To launch the interactive dashboard:")
    print("  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main() 