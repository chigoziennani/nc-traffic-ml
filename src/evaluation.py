"""
Model Evaluation Module for NC Traffic Forecasting

This module provides comprehensive evaluation metrics and comparison
tools for assessing the performance of different forecasting models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Any
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation and comparison."""
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.results = {}
        self.comparison_results = {}
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         model_name: str = "model") -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Basic metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # Percentage error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # R-squared
        r2 = r2_score(y_true, y_pred)
        
        # Additional metrics
        mape_median = np.median(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Directional accuracy (for time series)
        directional_accuracy = self._calculate_directional_accuracy(y_true, y_pred)
        
        # Theil's U statistic
        theil_u = self._calculate_theil_u(y_true, y_pred)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'MAPE_Median': mape_median,
            'R2': r2,
            'Directional_Accuracy': directional_accuracy,
            'Theil_U': theil_u
        }
        
        self.results[model_name] = metrics
        
        logger.info(f"{model_name} Metrics - RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%, R²: {r2:.3f}")
        
        return metrics
    
    def _calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy for time series."""
        if len(y_true) < 2:
            return 0.0
        
        # Calculate actual and predicted changes
        actual_changes = np.diff(y_true)
        predicted_changes = np.diff(y_pred)
        
        # Count correct directional predictions
        correct_directions = np.sum(
            (actual_changes > 0) == (predicted_changes > 0)
        )
        
        return correct_directions / len(actual_changes)
    
    def _calculate_theil_u(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Theil's U statistic."""
        # Theil's U = sqrt(sum((y_pred - y_true)^2)) / sqrt(sum(y_true^2))
        numerator = np.sqrt(np.sum((y_pred - y_true) ** 2))
        denominator = np.sqrt(np.sum(y_true ** 2))
        
        if denominator == 0:
            return float('inf')
        
        return numerator / denominator
    
    def compare_models(self, results_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Compare multiple models and create a comparison table.
        
        Args:
            results_dict: Dictionary with model results
            
        Returns:
            DataFrame with model comparison
        """
        logger.info("Comparing model performances...")
        
        # Create comparison DataFrame
        comparison_data = []
        
        for model_name, metrics in results_dict.items():
            row = {'Model': model_name}
            row.update(metrics)
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by RMSE (lower is better)
        comparison_df = comparison_df.sort_values('RMSE')
        
        self.comparison_results = comparison_df
        
        logger.info("Model Comparison (sorted by RMSE):")
        logger.info(comparison_df[['Model', 'RMSE', 'MAE', 'MAPE', 'R2']].to_string(index=False))
        
        return comparison_df
    
    def create_evaluation_plots(self, y_true: np.ndarray, predictions_dict: Dict[str, np.ndarray],
                               save_path: str = None) -> None:
        """
        Create comprehensive evaluation plots.
        
        Args:
            y_true: True values
            predictions_dict: Dictionary with model predictions
            save_path: Path to save plots (optional)
        """
        logger.info("Creating evaluation plots...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Evaluation', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted Scatter Plot
        ax1 = axes[0, 0]
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (model_name, y_pred) in enumerate(predictions_dict.items()):
            color = colors[i % len(colors)]
            ax1.scatter(y_true, y_pred, alpha=0.6, label=model_name, color=color)
        
        # Perfect prediction line
        min_val = min(y_true.min(), min([pred.min() for pred in predictions_dict.values()]))
        max_val = max(y_true.max(), max([pred.max() for pred in predictions_dict.values()]))
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label='Perfect Prediction')
        
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Actual vs Predicted Values')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Residuals Plot
        ax2 = axes[0, 1]
        for i, (model_name, y_pred) in enumerate(predictions_dict.items()):
            color = colors[i % len(colors)]
            residuals = y_true - y_pred
            ax2.scatter(y_pred, residuals, alpha=0.6, label=model_name, color=color)
        
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.8)
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals Plot')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Error Distribution
        ax3 = axes[1, 0]
        for i, (model_name, y_pred) in enumerate(predictions_dict.items()):
            color = colors[i % len(colors)]
            errors = y_true - y_pred
            ax3.hist(errors, bins=30, alpha=0.6, label=model_name, color=color, density=True)
        
        ax3.set_xlabel('Prediction Errors')
        ax3.set_ylabel('Density')
        ax3.set_title('Error Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Metrics Comparison Bar Plot
        ax4 = axes[1, 1]
        if self.comparison_results is not None and not self.comparison_results.empty:
            metrics_to_plot = ['RMSE', 'MAE', 'MAPE']
            x = np.arange(len(metrics_to_plot))
            width = 0.8 / len(predictions_dict)
            
            for i, (model_name, y_pred) in enumerate(predictions_dict.items()):
                if model_name in self.results:
                    metrics = [self.results[model_name][metric] for metric in metrics_to_plot]
                    ax4.bar(x + i * width, metrics, width, label=model_name, alpha=0.8)
            
            ax4.set_xlabel('Metrics')
            ax4.set_ylabel('Values')
            ax4.set_title('Model Performance Metrics')
            ax4.set_xticks(x + width * (len(predictions_dict) - 1) / 2)
            ax4.set_xticklabels(metrics_to_plot)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Evaluation plots saved to {save_path}")
        
        plt.show()
    
    def create_time_series_plots(self, dates: pd.Series, y_true: np.ndarray, 
                                predictions_dict: Dict[str, np.ndarray],
                                segment_id: str = None, save_path: str = None) -> None:
        """
        Create time series comparison plots.
        
        Args:
            dates: Date series
            y_true: True values
            predictions_dict: Dictionary with model predictions
            segment_id: Highway segment ID (optional)
            save_path: Path to save plots (optional)
        """
        logger.info("Creating time series plots...")
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # 1. Full time series plot
        ax1 = axes[0]
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        # Plot actual values
        ax1.plot(dates, y_true, 'k-', linewidth=2, label='Actual', alpha=0.8)
        
        # Plot predictions
        for i, (model_name, y_pred) in enumerate(predictions_dict.items()):
            color = colors[i % len(colors)]
            ax1.plot(dates, y_pred, color=color, linewidth=1.5, label=model_name, alpha=0.7)
        
        ax1.set_xlabel('Date')
        ax1.set_ylabel('AADT')
        ax1.set_title(f'Traffic Volume Forecasts - {segment_id if segment_id else "All Segments"}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. Zoomed in view (last 30 days)
        ax2 = axes[1]
        if len(dates) > 30:
            recent_dates = dates[-30:]
            recent_true = y_true[-30:]
            recent_predictions = {name: pred[-30:] for name, pred in predictions_dict.items()}
            
            ax2.plot(recent_dates, recent_true, 'k-', linewidth=2, label='Actual', alpha=0.8)
            
            for i, (model_name, y_pred) in enumerate(recent_predictions.items()):
                color = colors[i % len(colors)]
                ax2.plot(recent_dates, y_pred, color=color, linewidth=1.5, label=model_name, alpha=0.7)
            
            ax2.set_xlabel('Date')
            ax2.set_ylabel('AADT')
            ax2.set_title('Recent 30 Days - Traffic Volume Forecasts')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Time series plots saved to {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, output_path: str = "evaluation_report.txt") -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Report content as string
        """
        logger.info("Generating evaluation report...")
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("NC TRAFFIC FORECASTING - MODEL EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Model comparison table
        if self.comparison_results is not None and not self.comparison_results.empty:
            report_lines.append("MODEL PERFORMANCE COMPARISON")
            report_lines.append("-" * 40)
            report_lines.append(self.comparison_results.to_string(index=False))
            report_lines.append("")
        
        # Detailed metrics for each model
        report_lines.append("DETAILED MODEL METRICS")
        report_lines.append("-" * 40)
        
        for model_name, metrics in self.results.items():
            report_lines.append(f"\n{model_name.upper()}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    report_lines.append(f"  {metric}: {value:.4f}")
                else:
                    report_lines.append(f"  {metric}: {value}")
        
        # Model ranking
        if self.comparison_results is not None and not self.comparison_results.empty:
            report_lines.append("\nMODEL RANKING (by RMSE)")
            report_lines.append("-" * 40)
            for i, (_, row) in enumerate(self.comparison_results.iterrows(), 1):
                report_lines.append(f"{i}. {row['Model']}: RMSE = {row['RMSE']:.2f}")
        
        # Recommendations
        report_lines.append("\nRECOMMENDATIONS")
        report_lines.append("-" * 40)
        
        if self.comparison_results is not None and not self.comparison_results.empty:
            best_model = self.comparison_results.iloc[0]['Model']
            report_lines.append(f"• Best performing model: {best_model}")
            
            # Check if LSTM is among the best
            lstm_models = [model for model in self.results.keys() if 'lstm' in model.lower()]
            if lstm_models:
                best_lstm = min(lstm_models, key=lambda x: self.results[x]['RMSE'])
                report_lines.append(f"• Best LSTM model: {best_lstm}")
            
            # Check if traditional ML is among the best
            traditional_models = [model for model in self.results.keys() if 'lstm' not in model.lower()]
            if traditional_models:
                best_traditional = min(traditional_models, key=lambda x: self.results[x]['RMSE'])
                report_lines.append(f"• Best traditional ML model: {best_traditional}")
        
        report_lines.append("\n" + "=" * 60)
        
        report_content = "\n".join(report_lines)
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Evaluation report saved to {output_path}")
        
        return report_content


def main():
    """Main function to demonstrate evaluation."""
    from data_processing import NCDataProcessor
    from traditional_ml import TraditionalMLModels
    from deep_learning import LSTMTrafficForecaster
    
    # Load data
    processor = NCDataProcessor()
    try:
        data = processor.load_data("nc_traffic_processed.csv", "processed")
    except FileNotFoundError:
        logger.info("Processed data not found. Running data processing first...")
        from data_processing import main as process_data
        process_data()
        data = processor.load_data("nc_traffic_processed.csv", "processed")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Example evaluation (you would typically load trained models)
    logger.info("This is a demonstration of the evaluation module.")
    logger.info("In practice, you would load trained models and evaluate them.")
    
    # Generate sample results for demonstration
    sample_true = np.random.normal(50000, 10000, 1000)
    sample_pred1 = sample_true + np.random.normal(0, 2000, 1000)  # Good prediction
    sample_pred2 = sample_true + np.random.normal(0, 5000, 1000)  # Worse prediction
    
    # Calculate metrics
    evaluator.calculate_metrics(sample_true, sample_pred1, "Model_A")
    evaluator.calculate_metrics(sample_true, sample_pred2, "Model_B")
    
    # Compare models
    comparison = evaluator.compare_models(evaluator.results)
    
    # Generate report
    report = evaluator.generate_evaluation_report()
    print(report)


if __name__ == "__main__":
    main() 