"""
Visualization Module for NC Traffic Forecasting

This module provides comprehensive visualization tools for traffic data analysis,
model performance comparison, and forecast visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Any, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TrafficVisualizer:
    """Comprehensive visualization tools for traffic data and forecasts."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size for matplotlib plots
        """
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
    def plot_traffic_trends(self, df: pd.DataFrame, segment_id: str = None,
                           save_path: str = None) -> None:
        """
        Plot traffic volume trends over time.
        
        Args:
            df: Traffic data DataFrame
            segment_id: Specific segment to plot (optional)
            save_path: Path to save plot (optional)
        """
        logger.info("Creating traffic trends plot...")
        
        if segment_id:
            data = df[df['segment_id'] == segment_id].copy()
            title = f"Traffic Volume Trends - {data['segment_name'].iloc[0]}"
        else:
            data = df.copy()
            title = "Traffic Volume Trends - All Segments"
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Daily traffic over time
        ax1 = axes[0, 0]
        if segment_id:
            ax1.plot(data['date'], data['aadt'], linewidth=1, alpha=0.7)
        else:
            for seg_id in data['segment_id'].unique():
                seg_data = data[data['segment_id'] == seg_id]
                ax1.plot(seg_data['date'], seg_data['aadt'], linewidth=1, alpha=0.7, 
                        label=seg_data['segment_name'].iloc[0])
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax1.set_xlabel('Date')
        ax1.set_ylabel('AADT')
        ax1.set_title('Daily Traffic Volume')
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. Monthly averages
        ax2 = axes[0, 1]
        monthly_data = data.groupby(['year', 'month'])['aadt'].mean().reset_index()
        monthly_data['date'] = pd.to_datetime(monthly_data[['year', 'month']].assign(day=1))
        
        if segment_id:
            ax2.plot(monthly_data['date'], monthly_data['aadt'], marker='o', linewidth=2)
        else:
            for seg_id in data['segment_id'].unique():
                seg_data = data[data['segment_id'] == seg_id]
                seg_monthly = seg_data.groupby(['year', 'month'])['aadt'].mean().reset_index()
                seg_monthly['date'] = pd.to_datetime(seg_monthly[['year', 'month']].assign(day=1))
                ax2.plot(seg_monthly['date'], seg_monthly['aadt'], marker='o', linewidth=2, 
                        label=seg_data['segment_name'].iloc[0])
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Average AADT')
        ax2.set_title('Monthly Average Traffic Volume')
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Seasonal patterns
        ax3 = axes[1, 0]
        seasonal_data = data.groupby('month')['aadt'].mean()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        ax3.bar(range(1, 13), seasonal_data.values, color=self.colors[0], alpha=0.7)
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Average AADT')
        ax3.set_title('Seasonal Traffic Patterns')
        ax3.set_xticks(range(1, 13))
        ax3.set_xticklabels(months)
        ax3.grid(True, alpha=0.3)
        
        # 4. Weekly patterns
        ax4 = axes[1, 1]
        weekly_data = data.groupby('day_of_week')['aadt'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        ax4.bar(range(7), weekly_data.values, color=self.colors[1], alpha=0.7)
        ax4.set_xlabel('Day of Week')
        ax4.set_ylabel('Average AADT')
        ax4.set_title('Weekly Traffic Patterns')
        ax4.set_xticks(range(7))
        ax4.set_xticklabels(days)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Traffic trends plot saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, results_dict: Dict[str, Dict[str, float]],
                             save_path: str = None) -> None:
        """
        Create comprehensive model comparison plots.
        
        Args:
            results_dict: Dictionary with model results
            save_path: Path to save plot (optional)
        """
        logger.info("Creating model comparison plots...")
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Extract metrics
        models = list(results_dict.keys())
        rmse_values = [results_dict[model]['RMSE'] for model in models]
        mae_values = [results_dict[model]['MAE'] for model in models]
        mape_values = [results_dict[model]['MAPE'] for model in models]
        r2_values = [results_dict[model]['R2'] for model in models]
        
        # 1. RMSE comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(models, rmse_values, color=self.colors[0], alpha=0.7)
        ax1.set_ylabel('RMSE')
        ax1.set_title('Root Mean Square Error')
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, rmse_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_values)*0.01,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # 2. MAE comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(models, mae_values, color=self.colors[1], alpha=0.7)
        ax2.set_ylabel('MAE')
        ax2.set_title('Mean Absolute Error')
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        for bar, value in zip(bars2, mae_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_values)*0.01,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # 3. MAPE comparison
        ax3 = axes[1, 0]
        bars3 = ax3.bar(models, mape_values, color=self.colors[2], alpha=0.7)
        ax3.set_ylabel('MAPE (%)')
        ax3.set_title('Mean Absolute Percentage Error')
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        for bar, value in zip(bars3, mape_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mape_values)*0.01,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # 4. R² comparison
        ax4 = axes[1, 1]
        bars4 = ax4.bar(models, r2_values, color=self.colors[3], alpha=0.7)
        ax4.set_ylabel('R²')
        ax4.set_title('Coefficient of Determination')
        ax4.grid(True, alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        for bar, value in zip(bars4, r2_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(r2_values)*0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_forecasts(self, historical_df: pd.DataFrame, forecast_df: pd.DataFrame,
                      segment_id: str = None, save_path: str = None) -> None:
        """
        Plot historical data and forecasts.
        
        Args:
            historical_df: Historical traffic data
            forecast_df: Forecast data
            segment_id: Specific segment to plot (optional)
            save_path: Path to save plot (optional)
        """
        logger.info("Creating forecast plots...")
        
        # Filter data for specific segment if provided
        if segment_id:
            hist_data = historical_df[historical_df['segment_id'] == segment_id].copy()
            fore_data = forecast_df[forecast_df['segment_id'] == segment_id].copy()
            title = f"Traffic Forecasts - {hist_data['segment_name'].iloc[0]}"
        else:
            hist_data = historical_df.copy()
            fore_data = forecast_df.copy()
            title = "Traffic Forecasts - All Segments"
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Full timeline with forecasts
        ax1 = axes[0]
        
        # Plot historical data
        if segment_id:
            ax1.plot(hist_data['date'], hist_data['aadt'], 'k-', linewidth=2, 
                    label='Historical', alpha=0.8)
        else:
            for seg_id in hist_data['segment_id'].unique():
                seg_data = hist_data[hist_data['segment_id'] == seg_id]
                ax1.plot(seg_data['date'], seg_data['aadt'], linewidth=1.5, 
                        label=f"{seg_data['segment_name'].iloc[0]} (Historical)", alpha=0.7)
        
        # Plot forecasts
        forecast_cols = [col for col in fore_data.columns if 'prediction' in col]
        colors = self.colors[:len(forecast_cols)]
        
        for i, col in enumerate(forecast_cols):
            if segment_id:
                ax1.plot(fore_data['date'], fore_data[col], color=colors[i], 
                        linewidth=2, linestyle='--', label=col.replace('_prediction', '').title(), alpha=0.8)
            else:
                for seg_id in fore_data['segment_id'].unique():
                    seg_fore = fore_data[fore_data['segment_id'] == seg_id]
                    ax1.plot(seg_fore['date'], seg_fore[col], color=colors[i], 
                            linewidth=1.5, linestyle='--', 
                            label=f"{seg_fore['segment_name'].iloc[0]} ({col.replace('_prediction', '').title()})", alpha=0.7)
        
        ax1.set_xlabel('Date')
        ax1.set_ylabel('AADT')
        ax1.set_title('Historical Data and Forecasts')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. Forecast comparison (last 30 days of historical + forecasts)
        ax2 = axes[1]
        
        # Get last 30 days of historical data
        last_30_days = hist_data.sort_values('date').tail(30)
        
        if segment_id:
            ax2.plot(last_30_days['date'], last_30_days['aadt'], 'k-', linewidth=2, 
                    label='Historical (Last 30 Days)', alpha=0.8)
        else:
            for seg_id in last_30_days['segment_id'].unique():
                seg_data = last_30_days[last_30_days['segment_id'] == seg_id]
                ax2.plot(seg_data['date'], seg_data['aadt'], linewidth=1.5, 
                        label=f"{seg_data['segment_name'].iloc[0]} (Historical)", alpha=0.7)
        
        # Plot forecasts
        for i, col in enumerate(forecast_cols):
            if segment_id:
                ax2.plot(fore_data['date'], fore_data[col], color=colors[i], 
                        linewidth=2, linestyle='--', label=col.replace('_prediction', '').title(), alpha=0.8)
            else:
                for seg_id in fore_data['segment_id'].unique():
                    seg_fore = fore_data[fore_data['segment_id'] == seg_id]
                    ax2.plot(seg_fore['date'], seg_fore[col], color=colors[i], 
                            linewidth=1.5, linestyle='--', 
                            label=f"{seg_fore['segment_name'].iloc[0]} ({col.replace('_prediction', '').title()})", alpha=0.7)
        
        ax2.set_xlabel('Date')
        ax2.set_ylabel('AADT')
        ax2.set_title('Recent Data and Forecasts')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Forecast plot saved to {save_path}")
        
        plt.show()
    
    def create_interactive_plot(self, df: pd.DataFrame, forecast_df: pd.DataFrame = None) -> go.Figure:
        """
        Create an interactive Plotly visualization.
        
        Args:
            df: Historical traffic data
            forecast_df: Forecast data (optional)
            
        Returns:
            Plotly figure object
        """
        logger.info("Creating interactive plot...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Traffic Volume Trends', 'Seasonal Patterns', 
                          'Weekly Patterns', 'Forecasts'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Traffic volume trends
        for segment_id in df['segment_id'].unique():
            seg_data = df[df['segment_id'] == segment_id]
            fig.add_trace(
                go.Scatter(
                    x=seg_data['date'],
                    y=seg_data['aadt'],
                    mode='lines',
                    name=seg_data['segment_name'].iloc[0],
                    line=dict(width=1)
                ),
                row=1, col=1
            )
        
        # 2. Seasonal patterns
        monthly_data = df.groupby('month')['aadt'].mean().reset_index()
        fig.add_trace(
            go.Bar(
                x=monthly_data['month'],
                y=monthly_data['aadt'],
                name='Monthly Average',
                marker_color='lightblue'
            ),
            row=1, col=2
        )
        
        # 3. Weekly patterns
        weekly_data = df.groupby('day_of_week')['aadt'].mean().reset_index()
        fig.add_trace(
            go.Bar(
                x=weekly_data['day_of_week'],
                y=weekly_data['aadt'],
                name='Weekly Average',
                marker_color='lightgreen'
            ),
            row=2, col=1
        )
        
        # 4. Forecasts (if available)
        if forecast_df is not None:
            for segment_id in forecast_df['segment_id'].unique():
                seg_fore = forecast_df[forecast_df['segment_id'] == segment_id]
                forecast_cols = [col for col in seg_fore.columns if 'prediction' in col]
                
                for col in forecast_cols:
                    fig.add_trace(
                        go.Scatter(
                            x=seg_fore['date'],
                            y=seg_fore[col],
                            mode='lines',
                            name=f"{seg_fore['segment_name'].iloc[0]} ({col.replace('_prediction', '').title()})",
                            line=dict(dash='dash')
                        ),
                        row=2, col=2
                    )
        
        # Update layout
        fig.update_layout(
            title_text="NC Traffic Data Analysis",
            showlegend=True,
            height=800
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="AADT", row=1, col=1)
        fig.update_xaxes(title_text="Month", row=1, col=2)
        fig.update_yaxes(title_text="Average AADT", row=1, col=2)
        fig.update_xaxes(title_text="Day of Week", row=2, col=1)
        fig.update_yaxes(title_text="Average AADT", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_yaxes(title_text="AADT", row=2, col=2)
        
        return fig
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, 
                               top_n: int = 15, save_path: str = None) -> None:
        """
        Plot feature importance for tree-based models.
        
        Args:
            importance_df: DataFrame with feature importance scores
            top_n: Number of top features to display
            save_path: Path to save plot (optional)
        """
        logger.info("Creating feature importance plot...")
        
        # Get top N features
        top_features = importance_df.head(top_n)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Horizontal bar plot
        bars = plt.barh(range(len(top_features)), top_features['importance'], 
                       color=self.colors[0], alpha=0.7)
        
        # Customize plot
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Most Important Features')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, top_features['importance'])):
            plt.text(bar.get_width() + max(top_features['importance']) * 0.01, 
                    bar.get_y() + bar.get_height()/2, 
                    f'{value:.4f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_correlation_matrix(self, df: pd.DataFrame, save_path: str = None) -> None:
        """
        Plot correlation matrix of numerical features.
        
        Args:
            df: DataFrame with numerical features
            save_path: Path to save plot (optional)
        """
        logger.info("Creating correlation matrix plot...")
        
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numerical_cols].corr()
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation matrix plot saved to {save_path}")
        
        plt.show()


def main():
    """Main function to demonstrate visualization."""
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
    
    # Initialize visualizer
    visualizer = TrafficVisualizer()
    
    # Create traffic trends plot
    visualizer.plot_traffic_trends(data, save_path="traffic_trends.png")
    
    # Create correlation matrix
    visualizer.plot_correlation_matrix(data, save_path="correlation_matrix.png")
    
    logger.info("Visualization demonstration complete!")


if __name__ == "__main__":
    main() 