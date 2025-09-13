"""
Crisis Prediction System Web Application
Real-time market crisis prediction dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List

from crisis_predictor import CrisisPredictionSystem
from config import EVENT_CATEGORIES, SYSTEM_CONFIG

# Page configuration
st.set_page_config(
    page_title="Market Crisis Prediction System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size: 18px !important;
    }
    .risk-low {
        background-color: #28a745;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    .risk-medium {
        background-color: #ffc107;
        color: black;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    .risk-high {
        background-color: #ff6b35;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    .risk-critical {
        background-color: #dc3545;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = CrisisPredictionSystem()
    st.session_state.last_update = None
    st.session_state.prediction_history = []
    st.session_state.auto_refresh = False
    
    # Try to load pre-trained models
    try:
        success = st.session_state.predictor.load_models()
        if success:
            st.session_state.models_loaded = True
        else:
            st.session_state.models_loaded = False
    except Exception as e:
        st.session_state.models_loaded = False
        print(f"Error loading models: {e}")

def create_gauge_chart(value: float, title: str, ranges: List[float] = [0.2, 0.5, 0.7, 0.85]) -> go.Figure:
    """Create a gauge chart for probability visualization"""
    
    # Determine color based on value
    if value < ranges[0]:
        color = "green"
    elif value < ranges[1]:
        color = "yellow"
    elif value < ranges[2]:
        color = "orange"
    else:
        color = "red"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, ranges[0]*100], 'color': '#e8f5e9'},
                {'range': [ranges[0]*100, ranges[1]*100], 'color': '#fff9c4'},
                {'range': [ranges[1]*100, ranges[2]*100], 'color': '#ffe0b2'},
                {'range': [ranges[2]*100, 100], 'color': '#ffcdd2'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 85
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_risk_timeline_chart(history: List[Dict]) -> go.Figure:
    """Create a timeline chart of risk predictions"""
    if not history:
        return go.Figure()
    
    df = pd.DataFrame(history)
    
    fig = go.Figure()
    
    # Add 30-day prediction line
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['30_day_prob'],
        mode='lines+markers',
        name='30-Day Risk',
        line=dict(color='orange', width=3)
    ))
    
    # Add 6-month prediction line
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['6_month_prob'],
        mode='lines+markers',
        name='6-Month Risk',
        line=dict(color='red', width=3)
    ))
    
    # Add risk threshold lines
    fig.add_hline(y=0.2, line_dash="dash", line_color="green", annotation_text="Low Risk")
    fig.add_hline(y=0.5, line_dash="dash", line_color="yellow", annotation_text="Medium Risk")
    fig.add_hline(y=0.7, line_dash="dash", line_color="orange", annotation_text="High Risk")
    fig.add_hline(y=0.85, line_dash="dash", line_color="red", annotation_text="Critical Risk")
    
    fig.update_layout(
        title="Risk Prediction Timeline",
        xaxis_title="Time",
        yaxis_title="Crisis Probability",
        yaxis=dict(range=[0, 1]),
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_sector_heatmap(sector_data: pd.DataFrame) -> go.Figure:
    """Create a heatmap of sector performance"""
    if sector_data.empty:
        return go.Figure()
    
    # Reshape data for heatmap
    values = sector_data['5_day_return'].values.reshape(-1, 1)
    
    fig = go.Figure(data=go.Heatmap(
        z=values,
        x=['5-Day Return'],
        y=sector_data.index,
        colorscale='RdYlGn',
        zmid=0,
        text=[[f"{v:.1f}%" for v in values.flatten()]],
        texttemplate="%{text}",
        textfont={"size": 14},
        colorbar=dict(title="Return %")
    ))
    
    fig.update_layout(
        title="Sector Performance Heatmap",
        height=400
    )
    
    return fig

def display_historical_events():
    """Display historical crisis events"""
    st.subheader("📚 Historical Crisis Events Database")
    
    # Create event categories filter
    categories = list(EVENT_CATEGORIES.keys())
    selected_categories = st.multiselect(
        "Filter by Event Type",
        categories,
        default=categories
    )
    
    # Create events dataframe
    from data_collector import HistoricalEventLabeler
    labeler = HistoricalEventLabeler("")
    events = labeler.events
    
    # Filter events
    filtered_events = [
        e for e in events 
        if e['type'] in selected_categories
    ]
    
    # Convert to dataframe
    events_df = pd.DataFrame(filtered_events)
    events_df['start'] = pd.to_datetime(events_df['start'])
    events_df['end'] = pd.to_datetime(events_df['end'])
    events_df['duration_days'] = (events_df['end'] - events_df['start']).dt.days
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Events", len(filtered_events))
    with col2:
        st.metric("Avg Severity", f"{events_df['severity'].mean():.2f}")
    with col3:
        st.metric("Most Common Type", events_df['type'].mode()[0])
    with col4:
        st.metric("Avg Duration", f"{events_df['duration_days'].mean():.0f} days")
    
    # Display events table
    st.dataframe(
        events_df[['name', 'type', 'start', 'end', 'severity', 'duration_days']].sort_values('start', ascending=False),
        use_container_width=True
    )
    
    # Create timeline visualization
    fig = px.timeline(
        events_df,
        x_start="start",
        x_end="end",
        y="type",
        color="severity",
        hover_name="name",
        title="Crisis Events Timeline"
    )
    fig.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(fig, use_container_width=True)

def main():
    # Header
    st.title("🚨 Market Crisis Prediction System")
    st.markdown("**Real-time monitoring and prediction of market crashes, financial crises, and systemic shocks**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Auto-refresh toggle
        st.session_state.auto_refresh = st.checkbox(
            "Auto-refresh (every 5 min)",
            value=st.session_state.auto_refresh
        )
        
        # Manual refresh button
        if st.button("🔄 Refresh Now", type="primary", use_container_width=True):
            st.session_state.last_update = None
        
        st.divider()
        
        # Model status
        st.subheader("📊 Model Status")
        if st.session_state.predictor.is_trained:
            st.success("✅ Models Loaded")
        else:
            if st.button("🚀 Train Models", use_container_width=True):
                with st.spinner("Training models... This may take a few minutes."):
                    success = st.session_state.predictor.train_models()
                    if success:
                        st.success("✅ Models trained successfully!")
                    else:
                        st.error("❌ Model training failed")
        
        st.divider()
        
        # Information
        st.info("""
        **How it works:**
        - Analyzes 60+ years of market data
        - Monitors real-time indicators
        - Predicts crisis probability
        - Provides actionable recommendations
        """)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Live Predictions", "📊 Market Analysis", "📚 Historical Events", "🧪 Backtesting"])
    
    with tab1:
        # Get latest prediction
        current_time = datetime.now()
        
        # Check if we need to update
        prediction = None
        if (st.session_state.last_update is None or 
            (current_time - st.session_state.last_update).seconds > 300 or
            st.session_state.auto_refresh):
            
            with st.spinner("Analyzing market conditions..."):
                prediction = st.session_state.predictor.predict_live()
                
                if prediction:
                    st.session_state.last_update = current_time
                    st.session_state.last_prediction = prediction
                    
                    # Store in history
                    st.session_state.prediction_history.append({
                        'timestamp': prediction['timestamp'],
                        '30_day_prob': prediction['predictions']['30_day_probability'],
                        '6_month_prob': prediction['predictions']['6_month_probability'],
                        'risk_level': prediction['predictions']['risk_level']
                    })
                    
                    # Keep only last 100 predictions
                    if len(st.session_state.prediction_history) > 100:
                        st.session_state.prediction_history = st.session_state.prediction_history[-100:]
        
        # Use last prediction if available
        if not prediction and hasattr(st.session_state, 'last_prediction'):
            prediction = st.session_state.last_prediction
            
        if prediction:
            # Display update time
            st.caption(f"Last updated: {prediction['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Risk level display
            risk_level = prediction['predictions']['risk_level']
            risk_class = f"risk-{risk_level.lower()}"
            st.markdown(f'<div class="{risk_class}"><h2>Current Risk Level: {risk_level}</h2></div>', 
                       unsafe_allow_html=True)
            
            # Probability gauges
            col1, col2 = st.columns(2)
            
            with col1:
                fig_30 = create_gauge_chart(
                    prediction['predictions']['30_day_probability'],
                    "30-Day Crisis Probability"
                )
                st.plotly_chart(fig_30, use_container_width=True)
            
            with col2:
                fig_180 = create_gauge_chart(
                    prediction['predictions']['6_month_probability'],
                    "6-Month Crisis Probability"
                )
                st.plotly_chart(fig_180, use_container_width=True)
            
            # Risk factors
            st.subheader("⚠️ Risk Factors")
            if prediction['risk_assessment']['risk_factors']:
                for factor in prediction['risk_assessment']['risk_factors']:
                    severity_color = {
                        'LOW': '🟢',
                        'MEDIUM': '🟡',
                        'HIGH': '🔴',
                        'CRITICAL': '🚨'
                    }
                    st.write(f"{severity_color.get(factor['severity'], '⚪')} **{factor['factor']}**: {factor['value']}")
            else:
                st.success("No significant risk factors detected")
            
            # Recommendations
            st.subheader("💡 Recommendations")
            for rec in prediction['recommendations']:
                st.write(rec)
            
            # Prediction timeline
            if st.session_state.prediction_history:
                st.subheader("📈 Risk Evolution")
                fig_timeline = create_risk_timeline_chart(st.session_state.prediction_history)
                st.plotly_chart(fig_timeline, use_container_width=True)
    
    with tab2:
        st.subheader("🔍 Market Analysis")
        
        if prediction and 'market_indicators' in prediction:
            # Market indicators
            st.markdown("### Key Market Indicators")
            
            indicators = prediction['market_indicators']
            
            # Create columns for indicators
            cols = st.columns(4)
            col_idx = 0
            
            for key, value in indicators.items():
                with cols[col_idx % 4]:
                    # Handle different value types
                    if isinstance(value, (int, float)):
                        if key == 'VIX':
                            delta_color = "inverse" if value > 30 else "normal"
                            st.metric(key, f"{value:.2f}", delta_color=delta_color)
                        else:
                            st.metric(key, f"{value:.2f}")
                    else:
                        # For non-numeric values (like sector names)
                        st.metric(key, str(value))
                col_idx += 1
            
            # Sector performance
            if 'sector_performance' in st.session_state.predictor.data_collector.collect_all_data():
                st.markdown("### Sector Performance")
                sector_data = st.session_state.predictor.data_collector.get_sector_performance()
                if not sector_data.empty:
                    fig_sector = create_sector_heatmap(sector_data)
                    st.plotly_chart(fig_sector, use_container_width=True)
            
            # Most likely crisis type
            st.markdown("### Crisis Type Prediction")
            crisis_type = prediction['risk_assessment']['most_likely_event_type']
            if crisis_type in EVENT_CATEGORIES:
                st.info(f"**Most Likely Event Type**: {crisis_type}")
                st.write(f"**Description**: {EVENT_CATEGORIES[crisis_type]['description']}")
                st.write(f"**Historical Examples**: {', '.join(EVENT_CATEGORIES[crisis_type]['examples'][:3])}")
    
    with tab3:
        display_historical_events()
    
    with tab4:
        st.subheader("🧪 Backtesting Results")
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=pd.to_datetime("2010-01-01"),
                min_value=pd.to_datetime("1970-01-01"),
                max_value=pd.to_datetime("2023-12-31")
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=pd.to_datetime("2023-12-31"),
                min_value=start_date,
                max_value=pd.to_datetime("2023-12-31")
            )
        
        if st.button("Run Backtest", type="primary"):
            with st.spinner("Running backtest... This may take a few minutes."):
                results = st.session_state.predictor.backtest(
                    str(start_date), 
                    str(end_date)
                )
                
                if results:
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Events", results['total_events'])
                    with col2:
                        st.metric("Correctly Predicted", results['correctly_predicted'])
                    with col3:
                        st.metric("Accuracy", f"{results['accuracy']:.1%}")
                    
                    # Detailed results
                    if results['detailed_results']:
                        st.markdown("### Detailed Predictions")
                        results_df = pd.DataFrame(results['detailed_results'])
                        st.dataframe(results_df, use_container_width=True)
                    
                    # Performance by event type
                    if results['events_by_type']:
                        st.markdown("### Performance by Event Type")
                        type_df = pd.DataFrame(results['events_by_type']).T
                        type_df['accuracy'] = type_df['sum'] / type_df['count']
                        st.bar_chart(type_df['accuracy'])

if __name__ == "__main__":
    main()
