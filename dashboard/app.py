"""
Interactive Streamlit Dashboard for NC Traffic Forecasting

This dashboard provides an interactive interface for exploring traffic data,
model performance, and forecasts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from datetime import datetime, timedelta

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from src.data_processing import NCDataProcessor
from src.traditional_ml import TraditionalMLModels
from src.deep_learning import LSTMTrafficForecaster
from src.evaluation import ModelEvaluator
from src.visualization import TrafficVisualizer

# Page configuration
st.set_page_config(
    page_title="NC Traffic Forecasting Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load data with caching."""
    try:
        processor = NCDataProcessor()
        data = processor.load_data("nc_traffic_processed.csv", "processed")
        return data, processor
    except FileNotFoundError:
        st.error("Data not found. Please run the main pipeline first: `python main.py`")
        return None, None

@st.cache_data
def load_forecasts():
    """Load forecasts with caching."""
    try:
        forecasts = pd.read_csv("forecasts.csv")
        forecasts['date'] = pd.to_datetime(forecasts['date'])
        return forecasts
    except FileNotFoundError:
        return None

@st.cache_data
def load_evaluation_results():
    """Load evaluation results with caching."""
    try:
        with open("evaluation_report.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        return None

def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<h1 class="main-header">🚦 NC Traffic Forecasting Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    data, processor = load_data()
    if data is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    
    # Data overview
    st.sidebar.subheader("📈 Data Overview")
    st.sidebar.metric("Total Records", f"{len(data):,}")
    st.sidebar.metric("Highway Segments", data['segment_id'].nunique())
    st.sidebar.metric("Date Range", f"{data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}")
    
    # Segment selection
    segments = data['segment_id'].unique()
    selected_segment = st.sidebar.selectbox(
        "Select Highway Segment:",
        segments,
        format_func=lambda x: data[data['segment_id'] == x]['segment_name'].iloc[0]
    )
    
    # Date range selection
    st.sidebar.subheader("📅 Date Range")
    min_date = data['date'].min()
    max_date = data['date'].max()
    
    date_range = st.sidebar.date_input(
        "Select Date Range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter data based on selection
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_data = data[
            (data['segment_id'] == selected_segment) &
            (data['date'] >= pd.Timestamp(start_date)) &
            (data['date'] <= pd.Timestamp(end_date))
        ]
    else:
        filtered_data = data[data['segment_id'] == selected_segment]
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Traffic Analysis", 
        "🤖 Model Performance", 
        "🔮 Forecasts", 
        "📈 Interactive Plots",
        "📋 Reports"
    ])
    
    with tab1:
        st.header("📊 Traffic Volume Analysis")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average AADT", f"{filtered_data['aadt'].mean():,.0f}")
        with col2:
            st.metric("Max AADT", f"{filtered_data['aadt'].max():,.0f}")
        with col3:
            st.metric("Min AADT", f"{filtered_data['aadt'].min():,.0f}")
        with col4:
            st.metric("Std Dev", f"{filtered_data['aadt'].std():,.0f}")
        
        # Traffic trends
        st.subheader("Traffic Volume Trends")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered_data['date'],
            y=filtered_data['aadt'],
            mode='lines',
            name='AADT',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title=f"Traffic Volume - {filtered_data['segment_name'].iloc[0]}",
            xaxis_title="Date",
            yaxis_title="AADT",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal and weekly patterns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Seasonal Patterns")
            monthly_avg = filtered_data.groupby('month')['aadt'].mean()
            
            fig_monthly = px.bar(
                x=monthly_avg.index,
                y=monthly_avg.values,
                title="Monthly Average Traffic",
                labels={'x': 'Month', 'y': 'Average AADT'}
            )
            st.plotly_chart(fig_monthly, use_container_width=True)
        
        with col2:
            st.subheader("Weekly Patterns")
            weekly_avg = filtered_data.groupby('day_of_week')['aadt'].mean()
            
            fig_weekly = px.bar(
                x=weekly_avg.index,
                y=weekly_avg.values,
                title="Weekly Average Traffic",
                labels={'x': 'Day of Week', 'y': 'Average AADT'}
            )
            st.plotly_chart(fig_weekly, use_container_width=True)
    
    with tab2:
        st.header("🤖 Model Performance")
        
        # Load evaluation results
        evaluation_report = load_evaluation_results()
        
        if evaluation_report:
            st.subheader("Model Performance Summary")
            
            # Parse and display metrics
            lines = evaluation_report.split('\n')
            in_comparison = False
            comparison_data = []
            
            for line in lines:
                if "MODEL PERFORMANCE COMPARISON" in line:
                    in_comparison = True
                    continue
                elif in_comparison and line.strip() == "":
                    in_comparison = False
                    continue
                elif in_comparison and "---" not in line and line.strip():
                    comparison_data.append(line)
            
            if comparison_data:
                # Create a simple table
                st.write("**Model Comparison (sorted by RMSE):**")
                comparison_df = pd.DataFrame([line.split() for line in comparison_data[1:]], 
                                          columns=comparison_data[0].split())
                st.dataframe(comparison_df, use_container_width=True)
            
            # Display full report
            with st.expander("📋 Full Evaluation Report"):
                st.text(evaluation_report)
        else:
            st.warning("Evaluation report not found. Please run the main pipeline first.")
    
    with tab3:
        st.header("🔮 Traffic Forecasts")
        
        # Load forecasts
        forecasts = load_forecasts()
        
        if forecasts is not None:
            # Filter forecasts for selected segment
            segment_forecasts = forecasts[forecasts['segment_id'] == selected_segment]
            
            if not segment_forecasts.empty:
                st.subheader(f"30-Day Forecast - {segment_forecasts['segment_name'].iloc[0]}")
                
                # Get prediction columns
                prediction_cols = [col for col in segment_forecasts.columns if 'prediction' in col]
                
                # Create forecast plot
                fig = go.Figure()
                
                # Add historical data (last 30 days)
                historical_data = filtered_data.tail(30)
                fig.add_trace(go.Scatter(
                    x=historical_data['date'],
                    y=historical_data['aadt'],
                    mode='lines',
                    name='Historical (Last 30 Days)',
                    line=dict(color='black', width=3)
                ))
                
                # Add forecasts
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                for i, col in enumerate(prediction_cols):
                    fig.add_trace(go.Scatter(
                        x=segment_forecasts['date'],
                        y=segment_forecasts[col],
                        mode='lines',
                        name=col.replace('_prediction', '').title(),
                        line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                    ))
                
                fig.update_layout(
                    title="Traffic Volume Forecast",
                    xaxis_title="Date",
                    yaxis_title="AADT",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast table
                st.subheader("Forecast Values")
                forecast_display = segment_forecasts[['date'] + prediction_cols].copy()
                forecast_display['date'] = forecast_display['date'].dt.strftime('%Y-%m-%d')
                
                for col in prediction_cols:
                    forecast_display[col] = forecast_display[col].round(0).astype(int)
                
                st.dataframe(forecast_display, use_container_width=True)
                
            else:
                st.warning(f"No forecasts available for {selected_segment}")
        else:
            st.warning("Forecasts not found. Please run the main pipeline first.")
    
    with tab4:
        st.header("📈 Interactive Analysis")
        
        # Correlation analysis
        st.subheader("Feature Correlation Analysis")
        
        # Select numerical features for correlation
        numerical_cols = filtered_data.select_dtypes(include=[np.number]).columns.tolist()
        selected_features = st.multiselect(
            "Select features for correlation analysis:",
            numerical_cols,
            default=numerical_cols[:10]  # Default to first 10 features
        )
        
        if selected_features:
            corr_data = filtered_data[selected_features].corr()
            
            fig_corr = px.imshow(
                corr_data,
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Time series decomposition
        st.subheader("Time Series Analysis")
        
        # Rolling statistics
        window_size = st.slider("Rolling Window Size:", min_value=7, max_value=90, value=30)
        
        rolling_mean = filtered_data['aadt'].rolling(window=window_size).mean()
        rolling_std = filtered_data['aadt'].rolling(window=window_size).std()
        
        fig_rolling = go.Figure()
        
        fig_rolling.add_trace(go.Scatter(
            x=filtered_data['date'],
            y=filtered_data['aadt'],
            mode='lines',
            name='Original AADT',
            line=dict(color='lightblue', width=1)
        ))
        
        fig_rolling.add_trace(go.Scatter(
            x=filtered_data['date'],
            y=rolling_mean,
            mode='lines',
            name=f'{window_size}-Day Rolling Mean',
            line=dict(color='red', width=2)
        ))
        
        fig_rolling.add_trace(go.Scatter(
            x=filtered_data['date'],
            y=rolling_mean + rolling_std,
            mode='lines',
            name=f'{window_size}-Day Rolling Std (Upper)',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        fig_rolling.add_trace(go.Scatter(
            x=filtered_data['date'],
            y=rolling_mean - rolling_std,
            mode='lines',
            name=f'{window_size}-Day Rolling Std (Lower)',
            line=dict(color='gray', width=1, dash='dash'),
            fill='tonexty'
        ))
        
        fig_rolling.update_layout(
            title=f"Rolling Statistics (Window: {window_size} days)",
            xaxis_title="Date",
            yaxis_title="AADT",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_rolling, use_container_width=True)
    
    with tab5:
        st.header("📋 Reports & Documentation")
        
        # Data summary
        st.subheader("📊 Data Summary")
        
        summary = processor.get_data_summary(data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Information:**")
            st.write(f"- Total Records: {summary['total_records']:,}")
            st.write(f"- Highway Segments: {summary['segments']}")
            st.write(f"- Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        
        with col2:
            st.write("**Traffic Statistics:**")
            st.write(f"- Mean AADT: {summary['aadt_stats']['mean']:,.0f}")
            st.write(f"- Median AADT: {summary['aadt_stats']['median']:,.0f}")
            st.write(f"- Std Dev: {summary['aadt_stats']['std']:,.0f}")
        
        # Highway segments info
        st.subheader("🛣️ Highway Segments")
        
        segments_info = data.groupby(['segment_id', 'segment_name']).agg({
            'aadt': ['mean', 'std', 'min', 'max']
        }).round(0)
        
        segments_info.columns = ['Mean AADT', 'Std AADT', 'Min AADT', 'Max AADT']
        segments_info = segments_info.reset_index()
        
        st.dataframe(segments_info, use_container_width=True)
        
        # Download options
        st.subheader("💾 Download Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_data = filtered_data.to_csv(index=False)
            st.download_button(
                label="Download Filtered Data (CSV)",
                data=csv_data,
                file_name=f"traffic_data_{selected_segment}.csv",
                mime="text/csv"
            )
        
        with col2:
            if forecasts is not None:
                csv_forecasts = forecasts.to_csv(index=False)
                st.download_button(
                    label="Download Forecasts (CSV)",
                    data=csv_forecasts,
                    file_name="traffic_forecasts.csv",
                    mime="text/csv"
                )
        
        with col3:
            if evaluation_report:
                st.download_button(
                    label="Download Evaluation Report (TXT)",
                    data=evaluation_report,
                    file_name="evaluation_report.txt",
                    mime="text/plain"
                )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>🚦 NC Traffic Forecasting Dashboard | Built with Streamlit</p>
            <p>Data: Synthetic NC Highway Traffic Data | Models: Traditional ML + LSTM</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 