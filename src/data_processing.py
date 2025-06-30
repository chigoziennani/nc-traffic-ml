"""
Data Processing Module for NC Traffic Forecasting

This module handles data collection, preprocessing, and feature engineering
for North Carolina highway traffic volume forecasting.
"""

import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NCDataProcessor:
    """Data processor for NC traffic data collection and preprocessing."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data processor.
        
        Args:
            data_dir: Directory to store data files
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        
        # Create directories if they don't exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # NC Highway segments with realistic characteristics
        self.highway_segments = {
            'I-40_Raleigh_Durham': {
                'name': 'I-40 Raleigh-Durham',
                'base_aadt': 85000,
                'growth_rate': 0.025,
                'seasonal_factor': 0.15,
                'noise_level': 0.08
            },
            'I-85_Charlotte': {
                'name': 'I-85 Charlotte',
                'base_aadt': 95000,
                'growth_rate': 0.03,
                'seasonal_factor': 0.12,
                'noise_level': 0.07
            },
            'I-95_Fayetteville': {
                'name': 'I-95 Fayetteville',
                'base_aadt': 65000,
                'growth_rate': 0.02,
                'seasonal_factor': 0.18,
                'noise_level': 0.09
            },
            'US-421_Winston_Salem': {
                'name': 'US-421 Winston-Salem',
                'base_aadt': 45000,
                'growth_rate': 0.018,
                'seasonal_factor': 0.10,
                'noise_level': 0.06
            },
            'NC-147_Durham': {
                'name': 'NC-147 Durham',
                'base_aadt': 35000,
                'growth_rate': 0.035,
                'seasonal_factor': 0.08,
                'noise_level': 0.05
            }
        }
    
    def generate_synthetic_traffic_data(self, start_year: int = 2010, end_year: int = 2023) -> pd.DataFrame:
        """
        Generate synthetic traffic data for NC highways.
        
        Args:
            start_year: Start year for data generation
            end_year: End year for data generation
            
        Returns:
            DataFrame with synthetic traffic data
        """
        logger.info("Generating synthetic NC traffic data...")
        
        # Create date range
        dates = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq='D')
        
        data_list = []
        
        for segment_id, segment_info in self.highway_segments.items():
            logger.info(f"Generating data for {segment_info['name']}")
            
            for date in dates:
                # Base traffic volume
                years_from_start = (date.year - start_year)
                base_volume = segment_info['base_aadt'] * (1 + segment_info['growth_rate']) ** years_from_start
                
                # Seasonal variation (higher in summer, lower in winter)
                seasonal_effect = np.sin(2 * np.pi * date.dayofyear / 365.25) * segment_info['seasonal_factor']
                
                # Weekly pattern (weekends have different traffic)
                weekly_effect = 0.05 if date.weekday() >= 5 else 0  # 5% increase on weekends
                
                # Holiday effects (major holidays reduce traffic)
                holiday_effect = self._get_holiday_effect(date)
                
                # Random noise
                noise = np.random.normal(0, segment_info['noise_level'])
                
                # Calculate final AADT
                aadt = base_volume * (1 + seasonal_effect + weekly_effect + holiday_effect + noise)
                aadt = max(aadt, 1000)  # Ensure minimum traffic
                
                data_list.append({
                    'date': date,
                    'segment_id': segment_id,
                    'segment_name': segment_info['name'],
                    'aadt': int(aadt),
                    'year': date.year,
                    'month': date.month,
                    'day_of_year': date.dayofyear,
                    'day_of_week': date.weekday(),
                    'is_weekend': 1 if date.weekday() >= 5 else 0,
                    'is_holiday': 1 if holiday_effect < 0 else 0
                })
        
        df = pd.DataFrame(data_list)
        logger.info(f"Generated {len(df)} records for {len(self.highway_segments)} highway segments")
        
        return df
    
    def _get_holiday_effect(self, date: datetime) -> float:
        """Calculate holiday effect on traffic volume."""
        holidays = [
            (1, 1),   # New Year's Day
            (7, 4),   # Independence Day
            (11, 11), # Veterans Day
            (12, 25), # Christmas
            (12, 31), # New Year's Eve
        ]
        
        if (date.month, date.day) in holidays:
            return -0.15  # 15% reduction on major holidays
        
        # Thanksgiving (4th Thursday in November)
        if date.month == 11 and date.weekday() == 3:
            week_of_month = (date.day - 1) // 7 + 1
            if week_of_month == 4:
                return -0.20  # 20% reduction on Thanksgiving
        
        return 0.0
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the traffic data for modeling.
        
        Args:
            df: Raw traffic data DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Preprocessing traffic data...")
        
        # Create a copy to avoid modifying original data
        processed_df = df.copy()
        
        # Convert date to datetime if not already
        processed_df['date'] = pd.to_datetime(processed_df['date'])
        
        # Sort by segment and date
        processed_df = processed_df.sort_values(['segment_id', 'date']).reset_index(drop=True)
        
        # Add time-based features
        processed_df = self._add_time_features(processed_df)
        
        # Add lag features
        processed_df = self._add_lag_features(processed_df)
        
        # Add rolling statistics
        processed_df = self._add_rolling_features(processed_df)
        
        # Remove rows with NaN values (from lag features)
        processed_df = processed_df.dropna().reset_index(drop=True)
        
        logger.info(f"Preprocessing complete. Final dataset shape: {processed_df.shape}")
        
        return processed_df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features to the dataset."""
        # Extract time components
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['day_of_week'] = df['date'].dt.weekday
        df['quarter'] = df['date'].dt.quarter
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Cyclical encoding for periodic features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features for time series modeling."""
        # Group by segment to create lag features within each segment
        for segment_id in df['segment_id'].unique():
            segment_mask = df['segment_id'] == segment_id
            segment_data = df[segment_mask].copy()
            
            # Lag features (previous days)
            for lag in [1, 7, 14, 30]:
                df.loc[segment_mask, f'aadt_lag_{lag}'] = segment_data['aadt'].shift(lag)
            
            # Year-over-year lag (same day last year)
            df.loc[segment_mask, 'aadt_lag_365'] = segment_data['aadt'].shift(365)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistics features."""
        # Group by segment to create rolling features within each segment
        for segment_id in df['segment_id'].unique():
            segment_mask = df['segment_id'] == segment_id
            segment_data = df[segment_mask].copy()
            
            # Rolling means
            for window in [7, 14, 30, 90]:
                df.loc[segment_mask, f'aadt_rolling_mean_{window}'] = segment_data['aadt'].rolling(window=window, min_periods=1).mean()
                df.loc[segment_mask, f'aadt_rolling_std_{window}'] = segment_data['aadt'].rolling(window=window, min_periods=1).std()
            
            # Rolling min/max
            df.loc[segment_mask, 'aadt_rolling_min_30'] = segment_data['aadt'].rolling(window=30, min_periods=1).min()
            df.loc[segment_mask, 'aadt_rolling_max_30'] = segment_data['aadt'].rolling(window=30, min_periods=1).max()
        
        return df
    
    def save_data(self, df: pd.DataFrame, filename: str, directory: str = "processed") -> str:
        """
        Save processed data to file.
        
        Args:
            df: DataFrame to save
            filename: Name of the file
            directory: Directory to save in ('raw' or 'processed')
            
        Returns:
            Path to saved file
        """
        if directory == "raw":
            filepath = os.path.join(self.raw_dir, filename)
        else:
            filepath = os.path.join(self.processed_dir, filename)
        
        if filename.endswith('.csv'):
            df.to_csv(filepath, index=False)
        elif filename.endswith('.parquet'):
            df.to_parquet(filepath, index=False)
        else:
            raise ValueError("Unsupported file format. Use .csv or .parquet")
        
        logger.info(f"Data saved to {filepath}")
        return filepath
    
    def load_data(self, filename: str, directory: str = "processed") -> pd.DataFrame:
        """
        Load data from file.
        
        Args:
            filename: Name of the file to load
            directory: Directory to load from ('raw' or 'processed')
            
        Returns:
            Loaded DataFrame
        """
        if directory == "raw":
            filepath = os.path.join(self.raw_dir, filename)
        else:
            filepath = os.path.join(self.processed_dir, filename)
        
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filename.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        else:
            raise ValueError("Unsupported file format. Use .csv or .parquet")
        
        # Convert date column back to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        logger.info(f"Data loaded from {filepath}")
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for the dataset.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_records': len(df),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d')
            },
            'segments': df['segment_id'].nunique(),
            'segment_names': df['segment_name'].unique().tolist(),
            'aadt_stats': {
                'mean': df['aadt'].mean(),
                'std': df['aadt'].std(),
                'min': df['aadt'].min(),
                'max': df['aadt'].max(),
                'median': df['aadt'].median()
            },
            'missing_values': df.isnull().sum().to_dict()
        }
        
        return summary


def main():
    """Main function to run data processing pipeline."""
    # Initialize data processor
    processor = NCDataProcessor()
    
    # Generate synthetic data
    logger.info("Starting data generation...")
    raw_data = processor.generate_synthetic_traffic_data(start_year=2010, end_year=2023)
    
    # Save raw data
    processor.save_data(raw_data, "nc_traffic_raw.csv", "raw")
    
    # Preprocess data
    logger.info("Starting data preprocessing...")
    processed_data = processor.preprocess_data(raw_data)
    
    # Save processed data
    processor.save_data(processed_data, "nc_traffic_processed.csv", "processed")
    
    # Generate and display summary
    summary = processor.get_data_summary(processed_data)
    logger.info("Data processing complete!")
    logger.info(f"Dataset summary: {summary}")


if __name__ == "__main__":
    main() 