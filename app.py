import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta
import io
import shap

warnings.filterwarnings('ignore')

# ⚠️ CRITICAL: set_page_config MUST be the very first Streamlit command
st.set_page_config(
    page_title="Walmart Sales Forecasting Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Module Import and Initialization ---
# Try to import your existing modules from the src/ directory
# IMPORTANT: These imports now correctly reference the 'src' package
# We also import the WalmartSalesForecaster directly to manage data centrally
ADVANCED_ANALYSIS_AVAILABLE = False
PREDICTOR_AVAILABLE = False
forecaster_instance = None
analyzer_instance = None
predictor_instance = None

def fix_datetime_columns(df):
    """Fix datetime columns to ensure Arrow compatibility"""
    if df is None or df.empty:
        return df
    
    # Ensure Date column is properly formatted datetime64[ns]
    if 'Date' in df.columns:
        try:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Ensure it's datetime64[ns] specifically (not datetime64[ns, tz] or other variants)
            if df['Date'].dtype != 'datetime64[ns]':
                df['Date'] = df['Date'].astype('datetime64[ns]')
            
            # Remove any timezone info if present
            if hasattr(df['Date'].dtype, 'tz') and df['Date'].dtype.tz is not None:
                df['Date'] = df['Date'].dt.tz_localize(None)
                
        except Exception as e:
            st.error(f"Error fixing Date column: {e}")
            # Fallback: create a simple date range if conversion fails
            df['Date'] = pd.date_range(start='2011-01-01', periods=len(df), freq='W')
    
    return df

def create_dummy_data():
    """Create dummy data with proper datetime handling"""
    dummy_data = pd.DataFrame({
        'Store': [1, 2, 3, 4, 5] * 20,  # 100 records
        'Weekly_Sales': np.random.normal(500000, 100000, 100),
        'Temperature': np.random.normal(60, 15, 100),
        'Fuel_Price': np.random.normal(3.5, 0.5, 100),
        'CPI': np.random.normal(180, 10, 100),
        'Unemployment': np.random.normal(7, 2, 100),
        'IsHoliday': np.random.choice([0, 1], 100, p=[0.9, 0.1]),
        'Type': np.random.choice(['A', 'B', 'C'], 100),
        'Size': np.random.normal(150000, 50000, 100),
        'Dept': np.random.choice([1, 2, 3, 4, 5], 100),
        'Year': np.random.choice([2011, 2012, 2013], 100),
        'Month': np.random.choice(range(1, 13), 100),
        'Day': np.random.choice(range(1, 29), 100),
        'Week': np.random.choice(range(1, 53), 100),
        'DayOfWeek': np.random.choice(range(0, 7), 100)
    })
    
    # Create proper datetime column
    dummy_data['Date'] = pd.date_range(start='2011-01-01', periods=100, freq='W')
    
    # Ensure positive sales
    dummy_data['Weekly_Sales'] = np.abs(dummy_data['Weekly_Sales'])
    dummy_data['Size'] = np.abs(dummy_data['Size'])
    
    return fix_datetime_columns(dummy_data)

try:
    from src.main import WalmartSalesForecaster
    from src.advanced_analysis import AdvancedWalmartAnalyzer
    from src.predictor import WalmartPredictor

    # Use st.cache_resource for heavy objects like models/forecasters
    # This ensures data loading and model training happen only once
    @st.cache_resource(show_spinner="Loading and preparing data, training models...")
    def initialize_core_modules():
        st.info("Initializing WalmartSalesForecaster (loading data, preprocessing, training models)...")
        forecaster = WalmartSalesForecaster(data_path='data/Walmart.csv')
        if not forecaster.load_data():
            st.error("Failed to load Walmart sales data. Please ensure 'data/Walmart.csv' exists and is accessible.")
            return None, None, None
        forecaster.preprocess_data()
        if forecaster.processed_data is None or forecaster.processed_data.empty:
            st.error("Failed to preprocess Walmart sales data. Check data quality and date formats in your CSV.")
            return None, None, None
        
        # Fix datetime columns after preprocessing
        forecaster.processed_data = fix_datetime_columns(forecaster.processed_data)
        
        forecaster.train_models() # Train models here to make them available to predictor

        # Initialize Analyzer and Predictor, passing the already configured forecaster
        st.info("Initializing AdvancedWalmartAnalyzer...")
        analyzer = AdvancedWalmartAnalyzer()
        analyzer.forecaster = forecaster # Inject the pre-initialized forecaster
        if not analyzer.setup_data(): # This will now use data from the injected forecaster
            st.error("Failed to setup data for Advanced Analysis. Advanced features might be limited.")
            analyzer = None

        st.info("Initializing WalmartPredictor...")
        predictor = WalmartPredictor()
        predictor.forecaster = forecaster # Inject the pre-initialized forecaster
        if not predictor.setup_models(): # This will use models trained by the injected forecaster
            st.error("Failed to set up models for Prediction. Forecasting features might be limited.")
            predictor = None

        return forecaster, analyzer, predictor

    forecaster_instance, analyzer_instance, predictor_instance = initialize_core_modules()

    if forecaster_instance and forecaster_instance.processed_data is not None:
        df = fix_datetime_columns(forecaster_instance.processed_data.copy()) # Use the centrally processed data
        ADVANCED_ANALYSIS_AVAILABLE = (analyzer_instance is not None)
        PREDICTOR_AVAILABLE = (predictor_instance is not None and hasattr(predictor_instance, 'models_trained') and predictor_instance.models_trained)
    else:
        st.error("Critical: Data could not be loaded or processed. Please check your data file and logs.")
        # Fallback to dummy data if initial loading fails
        df = create_dummy_data()
        st.warning("Using sample data due to module import or initialization errors. Full functionality not available.")
        ADVANCED_ANALYSIS_AVAILABLE = False
        PREDICTOR_AVAILABLE = False

except ImportError as e:
    st.warning(f"⚠️ Missing `src` modules: {e}. Some advanced features and proper forecasting will be limited.")
    # Define fallback dummy classes if src modules cannot be imported at all
    class DummyWalmartSalesForecaster:
        def __init__(self, data_path=''):
            self.processed_data = create_dummy_data()
            self.models = {'RandomForest': None} # Dummy model
            self.label_encoders = {'Store': None, 'Dept': None, 'Type': None} # Dummy encoders
        def load_data(self): return True
        def preprocess_data(self): pass
        def train_models(self): pass
        def get_model(self, name): return None
        def prepare_single_prediction_data(self, store_id, date_str, holiday_flag, temperature, fuel_price, cpi, unemployment, dept_id, type_id):
            return pd.DataFrame([{'Store': store_id, 'Dept': dept_id, 'Temperature': temperature, 'Fuel_Price': fuel_price,
                                  'CPI': cpi, 'Unemployment': unemployment, 'IsHoliday': holiday_flag,
                                  'Year': pd.to_datetime(date_str).year, 'Month': pd.to_datetime(date_str).month,
                                  'Day': pd.to_datetime(date_str).day, 'Week': pd.to_datetime(date_str).isocalendar().week,
                                  'DayOfWeek': pd.to_datetime(date_str).dayofweek}])
    
    class DummyAdvancedWalmartAnalyzer:
        def __init__(self): self.forecaster = DummyWalmartSalesForecaster()
        def setup_data(self): return True
        def feature_importance_analysis(self): return pd.DataFrame()
        def hyperparameter_tuning(self): return None, None
        def advanced_models_comparison(self): return {}
        def sales_trend_analysis(self): return None
        def store_performance_analysis(self): return None
        def run_complete_advanced_analysis(self): return {}
    
    class DummyWalmartPredictor:
        def __init__(self): 
            self.forecaster = DummyWalmartSalesForecaster()
            self.models_trained = False
        def setup_models(self): 
            self.models_trained = True
            return True
        def predict_single_store(self, store_id, date_str, holiday_flag, temperature, fuel_price, cpi, unemployment, dept_id, type_id): 
            return 550000.0 + np.random.normal(0, 10000)
        def predict_multiple_stores(self, *args, **kwargs): 
            return {}
        def predict_time_series_for_store(self, store_id, start_date_str, num_weeks, temperature, fuel_price, cpi, unemployment, dept_id, type_id):
            dates = pd.date_range(start=pd.to_datetime(start_date_str), periods=num_weeks, freq='W')
            dummy_sales = np.linspace(500000, 700000, num_weeks) + np.random.normal(0, 20000, num_weeks)
            return dates.tolist(), dummy_sales.tolist()

    df = create_dummy_data()
    forecaster_instance = DummyWalmartSalesForecaster()
    analyzer_instance = DummyAdvancedWalmartAnalyzer()
    predictor_instance = DummyWalmartPredictor()
    ADVANCED_ANALYSIS_AVAILABLE = False
    PREDICTOR_AVAILABLE = False

# --- Weather & Google Trends Integration ---
try:
    from src.weather.weather_api import fetch_weather_for_all_cities
    from src.trends.google_trends import fetch_weekly_trends
    # Fetch weather and trends data for the full date range in your dataset
    start_date = '2010-01-01'
    end_date = '2012-12-31'
    weather_df = fetch_weather_for_all_cities(start_date, end_date)
    keywords = ['TV', 'groceries', 'electronics']
    trends_df = fetch_weekly_trends(keywords, start_date, end_date)
    # Merge into your main data if forecaster_instance is available
    if forecaster_instance and hasattr(forecaster_instance, 'merge_external_features'):
        forecaster_instance.merge_external_features(weather_df, trends_df)
        df = forecaster_instance.processed_data.copy()
        st.success('Weather and Google Trends features loaded and merged!')
except Exception as e:
    st.warning(f"Weather/Trends integration failed: {e}")

# Custom CSS for better styling
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
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stAlert > div {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def simple_forecast(store_data, forecast_weeks=12):
    """Simple forecasting method when predictor.py is not available"""
    if len(store_data) < 4:
        return pd.DataFrame(), "Insufficient data for forecasting (need at least 4 records)."

    store_data = store_data.sort_values('Date')
    recent_sales_data = store_data.tail(min(12, len(store_data)))

    if len(recent_sales_data) > 1:
        time_index = np.arange(len(recent_sales_data))
        try:
            trend_coeffs = np.polyfit(time_index, recent_sales_data['Weekly_Sales'], 1)
            trend = trend_coeffs[0]
            intercept = trend_coeffs[1]
        except np.linalg.LinAlgError:
            trend = 0
            intercept = recent_sales_data['Weekly_Sales'].iloc[0]
    else:
        trend = 0
        intercept = recent_sales_data['Weekly_Sales'].iloc[0] if not recent_sales_data.empty else 0

    last_date = store_data['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7),
                                 periods=forecast_weeks, freq='W')

    forecast_values = []
    for i in range(forecast_weeks):
        forecast_val = intercept + trend * (len(recent_sales_data) + i)
        week_of_year = future_dates[i].isocalendar()[1]
        seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * week_of_year / 52.14)
        noise_factor = np.random.normal(1, 0.03)

        final_forecast = forecast_val * seasonal_factor * noise_factor
        forecast_values.append(max(final_forecast, 100)) # Ensure sales are non-negative

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Sales': forecast_values
    })
    
    # Ensure proper datetime format
    forecast_df = fix_datetime_columns(forecast_df)
    return forecast_df, None

def validate_data_structure(data):
    """Validate the loaded data structure"""
    required_columns = ['Store', 'Date', 'Weekly_Sales']
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        st.error(f"❌ Missing required columns in processed data: {missing_columns}")
        st.info(f"Available columns: {list(data.columns)}")
        return False

    if not pd.api.types.is_datetime64_any_dtype(data['Date']):
        st.error(f"❌ Date column is not in datetime format after processing. Current type: {data['Date'].dtype}")
        return False

    if not pd.api.types.is_numeric_dtype(data['Weekly_Sales']):
        st.error("❌ Weekly_Sales column is not numeric after processing.")
        return False

    return True

def safe_get_store_info(data, store_id, column, default_value):
    """Safely get store information with fallback"""
    try:
        store_data = data[data['Store'] == store_id]
        if store_data.empty:
            st.warning(f"No data found for Store {store_id}")
            return default_value
        
        if column not in store_data.columns:
            st.warning(f"Column '{column}' not found in data")
            return default_value
            
        value = store_data[column].iloc[0]
        return value if pd.notna(value) else default_value
    except Exception as e:
        st.warning(f"Error getting {column} for Store {store_id}: {e}")
        return default_value

def show_feature_importance(forecaster_obj):
    st.subheader("🔑 Feature Importance (Random Forest)")
    try:
        importances = forecaster_obj.get_feature_importance()
        if importances:
            imp_df = pd.DataFrame(list(importances.items()), columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=False)
            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h", title="Feature Importances")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importances not available for the selected model.")
    except Exception as e:
        st.warning(f"Error displaying feature importance: {e}")

def show_anomalies(forecaster_obj):
    st.subheader("🚨 Anomaly Detection (Prediction Errors)")
    try:
        anomalies = forecaster_obj.detect_anomalies()
        if not anomalies.empty:
            st.warning(f"{len(anomalies)} anomalies detected (large prediction errors). See table below.")
            st.dataframe(anomalies, use_container_width=True)
        else:
            st.success("No significant anomalies detected.")
    except Exception as e:
        st.warning(f"Error running anomaly detection: {e}")

def show_scenario_analysis(forecaster_obj, store_id, dept_id, type_id):
    st.subheader("🧪 Scenario Analysis: What-if Simulation")
    st.write("Adjust weather and trend features to see predicted sales impact:")
    # Get feature ranges from training data
    X = forecaster_obj.X_train
    temp = st.slider("Temperature", float(X['Temperature'].min()), float(X['Temperature'].max()), float(X['Temperature'].mean())) if 'Temperature' in X.columns else 60.0
    fuel = st.slider("Fuel Price", float(X['Fuel_Price'].min()), float(X['Fuel_Price'].max()), float(X['Fuel_Price'].mean())) if 'Fuel_Price' in X.columns else 3.0
    cpi = st.slider("CPI", float(X['CPI'].min()), float(X['CPI'].max()), float(X['CPI'].mean())) if 'CPI' in X.columns else 180.0
    unemp = st.slider("Unemployment", float(X['Unemployment'].min()), float(X['Unemployment'].max()), float(X['Unemployment'].mean())) if 'Unemployment' in X.columns else 7.0
    date = st.date_input("Date", datetime.today())
    is_holiday = st.checkbox("Is Holiday", False)
    # Trends sliders
    trend_features = [col for col in X.columns if col.startswith('Trends_')]
    trend_vals = {}
    for col in trend_features:
        trend_vals[col] = st.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))
    # Prepare prediction
    pred = forecaster_obj.predict_sales(
        store_id=store_id,
        date_str=str(date),
        holiday_flag=is_holiday,
        temperature=temp,
        fuel_price=fuel,
        cpi=cpi,
        unemployment=unemp,
        dept_id=dept_id,
        type_id=type_id
    )
    st.metric("Predicted Sales", f"${pred:,.2f}" if pred is not None else "N/A")

def show_dynamic_filters(data):
    st.sidebar.subheader("🔎 Filter Data")
    city = st.sidebar.selectbox("City", ["All"] + sorted([c for c in data['City'].unique() if pd.notna(c)])) if 'City' in data.columns else None
    store = st.sidebar.selectbox("Store", ["All"] + sorted([int(s) for s in data['Store'].unique()]))
    dept = st.sidebar.selectbox("Department", ["All"] + sorted([int(d) for d in data['Dept'].unique()]))
    date_range = st.sidebar.date_input("Date Range", [data['Date'].min(), data['Date'].max()])
    df_filtered = data.copy()
    if city and city != "All":
        df_filtered = df_filtered[df_filtered['City'] == city]
    if store and store != "All":
        df_filtered = df_filtered[df_filtered['Store'] == int(store)]
    if dept and dept != "All":
        df_filtered = df_filtered[df_filtered['Dept'] == int(dept)]
    if date_range:
        df_filtered = df_filtered[(df_filtered['Date'] >= pd.to_datetime(date_range[0])) & (df_filtered['Date'] <= pd.to_datetime(date_range[1]))]
    return df_filtered

# --- Integrate new features into main() ---
def main():
    st.markdown('<h1 class="main-header">🛒 Walmart Sales Forecasting Dashboard</h1>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if ADVANCED_ANALYSIS_AVAILABLE:
            st.success("✅ Advanced Analysis Available")
        else:
            st.warning("⚠️ Advanced Analysis Limited")

    with col2:
        if PREDICTOR_AVAILABLE:
            st.success("✅ Predictor Available")
        else:
            st.warning("⚠️ Using Simple Forecasting")

    with col3:
        st.info("🚀 Dashboard Running")

    with col4:
        st.metric("Data Records", f"{len(df):,}" if df is not None else "0")

    # `df` is already set globally by the initialization block
    if df is None or df.empty:
        st.error("❌ Data could not be loaded or generated. Please check console for errors or ensure data/Walmart.csv is correct.")
        return

    if not validate_data_structure(df):
        st.error("❌ Processed data validation failed. Please check your data format and preprocessing in main.py.")
        return

    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["📊 Overview", "📈 Sales Analysis", "🔮 Forecasting", "🏪 Store Performance", "📅 Seasonal Trends", "🎯 Insights & Recommendations"]
    )

    df_filtered = show_dynamic_filters(df)

    if page == "📊 Overview":
        show_overview(df_filtered)
        show_feature_importance(forecaster_instance)
        show_anomalies(forecaster_instance)
    elif page == "📈 Sales Analysis":
        show_sales_analysis(df_filtered)
    elif page == "🔮 Forecasting":
        show_forecasting(df_filtered, predictor_instance, forecaster_instance)
        # Scenario analysis for selected store
        st.divider()
        st.header("🧪 Scenario Analysis")
        store_id = st.sidebar.number_input("Store ID for Scenario", min_value=1, value=1)
        dept_id = st.sidebar.number_input("Department ID for Scenario", min_value=1, value=1)
        type_id = st.sidebar.text_input("Type for Scenario", value='A')
        show_scenario_analysis(forecaster_instance, store_id, dept_id, type_id)
    elif page == "🏪 Store Performance":
        show_store_performance(df_filtered)
    elif page == "📅 Seasonal Trends":
        show_seasonal_trends(df_filtered)
    elif page == "🎯 Insights & Recommendations":
        show_insights(df_filtered, analyzer_instance)

def show_overview(data):
    st.header("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(data):,}")
    with col2:
        st.metric("Unique Stores", data['Store'].nunique())
    with col3:
        if 'Date' in data.columns and not data['Date'].empty:
            st.metric("Date Range", f"{data['Date'].dt.year.min()}-{data['Date'].dt.year.max()}")
        else:
            st.metric("Date Range", "N/A")
    with col4:
        st.metric("Total Sales", f"${data['Weekly_Sales'].sum():,.0f}")

    st.subheader("📋 Data Quality Check")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Missing Values", data.isnull().sum().sum())
    with col2:
        st.metric("Duplicate Records", data.duplicated().sum())
    with col3:
        st.metric("Data Completeness", f"{(1 - data.isnull().sum().sum() / data.size) * 100:.1f}%")

    st.subheader("📋 Data Sample")
    # Create a display version of the data with properly formatted dates
    display_data = data.head(10).copy()
    if 'Date' in display_data.columns:
        display_data['Date'] = display_data['Date'].dt.strftime('%Y-%m-%d')
    st.dataframe(display_data, use_container_width=True)

    st.subheader("📈 Statistical Summary")
    # Only show numeric columns in describe
    numeric_data = data.select_dtypes(include=[np.number])
    st.dataframe(numeric_data.describe(), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(data, x='Weekly_Sales', title='Weekly Sales Distribution',
                           nbins=50, color_discrete_sequence=['#1f77b4'])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(data, y='Weekly_Sales', title='Weekly Sales Box Plot',
                     color_discrete_sequence=['#ff7f0e'])
        st.plotly_chart(fig, use_container_width=True)

def show_sales_analysis(data):
    st.header("📈 Sales Analysis")

    st.subheader("📊 Sales Trends Over Time")

    col1, col2 = st.columns(2)
    with col1:
        agg_level = st.selectbox("Aggregation Level:", ["Weekly", "Monthly", "Quarterly"])
    with col2:
        unique_stores = sorted(data['Store'].unique())
        selected_stores = st.multiselect(
            "Select Stores (leave empty for all):",
            unique_stores,
            default=[]
        )

    if selected_stores:
        filtered_data = data[data['Store'].isin(selected_stores)].copy()
    else:
        filtered_data = data.copy()

    if filtered_data.empty:
        st.warning("No data to display for the selected filters.")
        return

    try:
        if agg_level == "Monthly":
            data_agg = filtered_data.groupby([filtered_data['Date'].dt.to_period('M')])['Weekly_Sales'].sum().reset_index()
            data_agg['Date'] = data_agg['Date'].dt.to_timestamp()
        elif agg_level == "Quarterly":
            data_agg = filtered_data.groupby([filtered_data['Date'].dt.to_period('Q')])['Weekly_Sales'].sum().reset_index()
            data_agg['Date'] = data_agg['Date'].dt.to_timestamp()
        else: # Weekly
            data_agg = filtered_data.groupby('Date')['Weekly_Sales'].sum().reset_index()

        data_agg = fix_datetime_columns(data_agg)
        
        fig = px.line(data_agg, x='Date', y='Weekly_Sales',
                      title=f'{agg_level} Sales Trends')
        fig.update_traces(line_color='#1f77b4', line_width=2)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating sales trend chart: {e}")

    st.subheader("🏪 Store Performance Comparison")

    col1, col2 = st.columns(2)
    with col1:
        top_n = st.slider("Number of top stores to display:", 5, 20, 10)
    with col2:
        metric_type = st.selectbox("Metric:", ["Total Sales", "Average Sales"])

    try:
        if metric_type == "Total Sales":
            store_sales = data.groupby('Store')['Weekly_Sales'].sum().sort_values(ascending=False).head(top_n)
        else:
            store_sales = data.groupby('Store')['Weekly_Sales'].mean().sort_values(ascending=False).head(top_n)

        fig = px.bar(x=store_sales.index, y=store_sales.values,
                     title=f'Top {top_n} Stores by {metric_type}',
                     color=store_sales.values,
                     color_continuous_scale='viridis')
        fig.update_xaxes(title="Store ID")
        fig.update_yaxes(title=f"{metric_type} ($)")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating store performance chart: {e}")

def show_forecasting(data, predictor, forecaster_obj):
    st.header("🔮 Sales Forecasting")

    col1, col2 = st.columns(2)
    with col1:
        selected_store = st.selectbox("Select Store for Forecasting:", sorted(data['Store'].unique()))
    with col2:
        forecast_weeks = st.slider("Forecast Period (weeks):", 1, 52, 12)

    store_data = data[data['Store'] == selected_store].copy()
    store_data = store_data.sort_values('Date')

    if store_data.empty:
        st.error("No data available for selected store.")
        return

    st.subheader(f"📊 Store {selected_store} Information")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(store_data))
    with col2:
        st.metric("Avg Weekly Sales", f"${store_data['Weekly_Sales'].mean():,.0f}")
    with col3:
        st.metric("Max Weekly Sales", f"${store_data['Weekly_Sales'].max():,.0f}")
    with col4:
        st.metric("Sales Volatility", f"{store_data['Weekly_Sales'].std():,.0f}")

    if st.button("🚀 Generate Forecast", type="primary"):
        try:
            with st.spinner('Generating forecast...'):
                forecast_df = pd.DataFrame()
                error = None

                last_historical_date = store_data['Date'].max()
                forecast_start_date = last_historical_date + pd.Timedelta(days=7)

                # Get average values for prediction features
                avg_fuel_price = safe_get_store_info(store_data, selected_store, 'Fuel_Price', 3.5)
                avg_temperature = safe_get_store_info(store_data, selected_store, 'Temperature', 60.0)
                avg_cpi = safe_get_store_info(store_data, selected_store, 'CPI', 180.0)
                avg_unemployment = safe_get_store_info(store_data, selected_store, 'Unemployment', 7.0)

                # Safe handling of store type and department
                store_type_val = safe_get_store_info(data, selected_store, 'Type', 'A')
                dept_id_val = safe_get_store_info(data, selected_store, 'Dept', 1)

                # Try to handle label encoding properly
                if (forecaster_obj and hasattr(forecaster_obj, 'label_encoders') and 
                    'Type' in forecaster_obj.label_encoders and 
                    forecaster_obj.label_encoders['Type'] is not None and
                    pd.api.types.is_numeric_dtype(type(store_type_val))):
                    try:
                        store_type_val = forecaster_obj.label_encoders['Type'].inverse_transform(np.array([store_type_val]))[0]
                    except Exception:
                        pass  # Use the original value if inverse transform fails

                if PREDICTOR_AVAILABLE and predictor and hasattr(predictor, 'models_trained') and predictor.models_trained:
                    try:
                        dates_list, predictions_list = predictor.predict_time_series_for_store(
                            store_id=selected_store,
                            start_date_str=forecast_start_date.strftime('%Y-%m-%d'),
                            num_weeks=forecast_weeks,
                            temperature=float(avg_temperature),
                            fuel_price=float(avg_fuel_price),
                            cpi=float(avg_cpi),
                            unemployment=float(avg_unemployment),
                            dept_id=int(dept_id_val),
                            type_id=store_type_val
                        )
                        forecast_df = pd.DataFrame({'Date': dates_list, 'Predicted_Sales': predictions_list})
                        forecast_df = fix_datetime_columns(forecast_df)
                        
                        if forecast_df.empty:
                            error = "Predictor returned no forecast data."

                    except Exception as e:
                        st.warning(f"Predictor failed: {str(e)}. Falling back to simple forecast.")
                        forecast_df, error = simple_forecast(store_data, forecast_weeks)
                else:
                    st.info("Using simple forecast as advanced predictor is not available or not trained.")
                    forecast_df, error = simple_forecast(store_data, forecast_weeks)

                if error:
                    st.error(f"Forecasting error: {error}")
                    return

                if forecast_df is None or forecast_df.empty:
                    st.error("Unable to generate forecast.")
                    return

                # Create combined plot
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=store_data['Date'],
                    y=store_data['Weekly_Sales'],
                    mode='lines+markers',
                    name='Historical Sales',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=4)
                ))

                fig.add_trace(go.Scatter(
                    x=forecast_df['Date'],
                    y=forecast_df['Predicted_Sales'],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='#ff7f0e', dash='dash', width=2),
                    marker=dict(size=4)
                ))

                fig.update_layout(
                    title=f'Sales Forecast for Store {selected_store}',
                    xaxis_title='Date',
                    yaxis_title='Weekly Sales ($)',
                    hovermode='x unified',
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

                st.subheader("📋 Forecast Details")
                forecast_display = forecast_df.copy()
                forecast_display['Date'] = forecast_display['Date'].dt.strftime('%Y-%m-%d')
                forecast_display['Predicted_Sales'] = forecast_display['Predicted_Sales'].round(2)
                forecast_display['Predicted_Sales_Formatted'] = forecast_display['Predicted_Sales'].apply(lambda x: f"${x:,.2f}")
                st.dataframe(forecast_display[['Date', 'Predicted_Sales_Formatted']], use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Avg Forecast", f"${forecast_df['Predicted_Sales'].mean():,.0f}")
                with col2:
                    st.metric("Total Forecast", f"${forecast_df['Predicted_Sales'].sum():,.0f}")

        except Exception as e:
            st.error(f"Error generating forecast: {str(e)}")

    # --- Add to .gitignore for API and cache files ---
    # Add these lines to your .gitignore:
    # .pytrends-cache/
    # weather_cache.json
    # trends_cache.json
    # .env

    st.write("### Weather & Trends Features (if available)")
    if 'Avg_Temperature' in data.columns:
        st.metric("Avg Temperature", f"{data['Avg_Temperature'].mean():.1f} °F")
    if 'Precipitation' in data.columns:
        st.metric("Precipitation", f"{data['Precipitation'].mean():.2f} in")
    for col in data.columns:
        if col.startswith('Trends_'):
            st.metric(col, f"{data[col].mean():.1f}")

def show_store_performance(data):
    st.header("🏪 Store Performance Analysis")

    try:
        store_metrics = data.groupby('Store').agg({
            'Weekly_Sales': ['mean', 'sum', 'std', 'count'],
            'Date': ['min', 'max']
        }).round(2)

        store_metrics.columns = ['Avg_Sales', 'Total_Sales', 'Sales_Volatility', 'Records', 'First_Date', 'Last_Date']
        store_metrics = store_metrics.reset_index()

        # Define performance categories
        q1, q2, q3 = store_metrics['Total_Sales'].quantile([0.25, 0.5, 0.75])
        min_sales = store_metrics['Total_Sales'].min()
        max_sales = store_metrics['Total_Sales'].max()
        def categorize_performance(sales):
            if sales >= q3:
                return 'High'
            elif sales >= q2:
                return 'Medium'
            elif sales >= q1:
                return 'Low'
            else:
                return 'Very Low'

        store_metrics['Performance_Category'] = store_metrics['Total_Sales'].apply(categorize_performance)

        st.subheader("📊 Store Performance Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("High Performers", len(store_metrics[store_metrics['Performance_Category'] == 'High']))
        with col2:
            st.metric("Medium Performers", len(store_metrics[store_metrics['Performance_Category'] == 'Medium']))
        with col3:
            st.metric("Low Performers", len(store_metrics[store_metrics['Performance_Category'] == 'Low']))
        with col4:
            st.metric("Very Low Performers", len(store_metrics[store_metrics['Performance_Category'] == 'Very Low']))

        # Performance scatter plot
        fig = px.scatter(store_metrics, x='Avg_Sales', y='Total_Sales', 
                        color='Performance_Category',
                        size='Records',
                        hover_data=['Store', 'Sales_Volatility'],
                        title='Store Performance: Average vs Total Sales')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Top and Bottom performers
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🏆 Top 10 Performers")
            top_stores = store_metrics.nlargest(10, 'Total_Sales')[['Store', 'Total_Sales', 'Avg_Sales']]
            top_stores['Total_Sales'] = top_stores['Total_Sales'].apply(lambda x: f"${x:,.0f}")
            top_stores['Avg_Sales'] = top_stores['Avg_Sales'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(top_stores, use_container_width=True)

        with col2:
            st.subheader("📉 Bottom 10 Performers")
            bottom_stores = store_metrics.nsmallest(10, 'Total_Sales')[['Store', 'Total_Sales', 'Avg_Sales']]
            bottom_stores['Total_Sales'] = bottom_stores['Total_Sales'].apply(lambda x: f"${x:,.0f}")
            bottom_stores['Avg_Sales'] = bottom_stores['Avg_Sales'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(bottom_stores, use_container_width=True)

        # Store type analysis if available
        if 'Type' in data.columns:
            st.subheader("🏪 Performance by Store Type")
            type_performance = data.groupby('Type').agg({
                'Weekly_Sales': ['mean', 'sum', 'count']
            }).round(2)
            type_performance.columns = ['Avg_Sales', 'Total_Sales', 'Records']
            type_performance = type_performance.reset_index()

            fig = px.bar(type_performance, x='Type', y='Total_Sales',
                        title='Total Sales by Store Type',
                        color='Avg_Sales',
                        color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error in store performance analysis: {str(e)}")

def show_seasonal_trends(data):
    st.header("📅 Seasonal Trends Analysis")

    try:
        # Create time-based columns if they don't exist
        if 'Month' not in data.columns:
            data = data.copy()
            data['Month'] = data['Date'].dt.month
            data['Quarter'] = data['Date'].dt.quarter
            data['Year'] = data['Date'].dt.year
            data['Week'] = data['Date'].dt.isocalendar().week

        st.subheader("📊 Monthly Sales Patterns")
        monthly_sales = data.groupby('Month')['Weekly_Sales'].agg(['mean', 'sum']).reset_index()
        monthly_sales['Month_Name'] = monthly_sales['Month'].apply(
            lambda x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][x-1]
        )

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(monthly_sales, x='Month_Name', y='mean',
                        title='Average Monthly Sales',
                        color='mean',
                        color_continuous_scale='blues')
            fig.update_xaxes(title="Month")
            fig.update_yaxes(title="Average Sales ($)")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.line(monthly_sales, x='Month_Name', y='sum',
                         title='Total Monthly Sales',
                         markers=True)
            fig.update_traces(line_color='#ff7f0e', line_width=3)
            fig.update_xaxes(title="Month")
            fig.update_yaxes(title="Total Sales ($)")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("📈 Quarterly Trends")
        quarterly_sales = data.groupby(['Year', 'Quarter'])['Weekly_Sales'].sum().reset_index()
        quarterly_sales['Quarter_Label'] = quarterly_sales['Year'].astype(str) + '-Q' + quarterly_sales['Quarter'].astype(str)

        fig = px.bar(quarterly_sales, x='Quarter_Label', y='Weekly_Sales',
                    title='Quarterly Sales Trends',
                    color='Weekly_Sales',
                    color_continuous_scale='viridis')
        fig.update_xaxes(title="Quarter")
        fig.update_yaxes(title="Total Sales ($)")
        st.plotly_chart(fig, use_container_width=True)

        # Holiday analysis if IsHoliday column exists
        if 'IsHoliday' in data.columns:
            st.subheader("🎄 Holiday vs Non-Holiday Sales")
            holiday_sales = data.groupby('IsHoliday')['Weekly_Sales'].agg(['mean', 'sum', 'count']).reset_index()
            holiday_sales['IsHoliday'] = holiday_sales['IsHoliday'].map({0: 'Non-Holiday', 1: 'Holiday'})

            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(holiday_sales, x='IsHoliday', y='mean',
                            title='Average Sales: Holiday vs Non-Holiday',
                            color='IsHoliday',
                            color_discrete_map={'Holiday': '#ff6b6b', 'Non-Holiday': '#4ecdc4'})
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.pie(holiday_sales, values='sum', names='IsHoliday',
                            title='Total Sales Distribution',
                            color_discrete_map={'Holiday': '#ff6b6b', 'Non-Holiday': '#4ecdc4'})
                st.plotly_chart(fig, use_container_width=True)

        # Heatmap of sales by month and year
        if len(data['Year'].unique()) > 1:
            st.subheader("🔥 Sales Heatmap by Month and Year")
            heatmap_data = data.groupby(['Year', 'Month'])['Weekly_Sales'].sum().reset_index()
            heatmap_pivot = heatmap_data.pivot(index='Year', columns='Month', values='Weekly_Sales')

            fig = px.imshow(heatmap_pivot,
                           title='Sales Heatmap (Month vs Year)',
                           labels=dict(x="Month", y="Year", color="Sales"),
                           aspect="auto",
                           color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error in seasonal trends analysis: {str(e)}")

def show_insights(data, analyzer):
    st.header("🎯 Insights & Recommendations")

    st.subheader("📊 Key Performance Insights")

    try:
        # Basic insights from data
        total_sales = data['Weekly_Sales'].sum()
        avg_sales = data['Weekly_Sales'].mean()
        peak_sales = data['Weekly_Sales'].max()
        best_store = data.loc[data['Weekly_Sales'].idxmax(), 'Store']

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Total Sales Across All Stores:** ${total_sales:,.0f}")
            st.info(f"**Average Weekly Sales:** ${avg_sales:,.0f}")
            st.info(f"**Peak Weekly Sales:** ${peak_sales:,.0f}")
            st.info(f"**Best Performing Store:** Store {best_store}")

        with col2:
            # Growth analysis
            if 'Date' in data.columns and len(data) > 1:
                data_sorted = data.sort_values('Date')
                first_half = data_sorted.iloc[:len(data_sorted)//2]['Weekly_Sales'].mean()
                second_half = data_sorted.iloc[len(data_sorted)//2:]['Weekly_Sales'].mean()
                growth_rate = ((second_half - first_half) / first_half) * 100

                st.info(f"**Sales Growth Rate:** {growth_rate:+.1f}%")
                
                if growth_rate > 0:
                    st.success("📈 Positive growth trend detected!")
                else:
                    st.warning("📉 Declining sales trend detected!")

        st.subheader("💡 Strategic Recommendations")

        # Store performance recommendations
        store_performance = data.groupby('Store')['Weekly_Sales'].agg(['mean', 'sum']).reset_index()
        top_quartile = store_performance['sum'].quantile(0.75)
        bottom_quartile = store_performance['sum'].quantile(0.25)

        underperforming_stores = store_performance[store_performance['sum'] < bottom_quartile]['Store'].tolist()
        top_performing_stores = store_performance[store_performance['sum'] >= top_quartile]['Store'].tolist()

        col1, col2 = st.columns(2)
        with col1:
            st.success("**🏆 Leverage High Performers:**")
            st.write(f"- Stores {', '.join(map(str, top_performing_stores[:5]))} are top performers")
            st.write("- Analyze their best practices and replicate across network")
            st.write("- Consider expanding product lines in these locations")

        with col2:
            st.warning("**📈 Improve Underperformers:**")
            if underperforming_stores:
                st.write(f"- Stores {', '.join(map(str, underperforming_stores[:5]))} need attention")
                st.write("- Conduct market analysis and customer surveys")
                st.write("- Consider promotional campaigns or layout changes")

        # Seasonal recommendations
        if 'Month' in data.columns or 'Date' in data.columns:
            if 'Month' not in data.columns:
                data = data.copy()
                data['Month'] = data['Date'].dt.month

            monthly_avg = data.groupby('Month')['Weekly_Sales'].mean()
            peak_month = monthly_avg.idxmax()
            low_month = monthly_avg.idxmin()

            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            st.info("**📅 Seasonal Strategy:**")
            st.write(f"- Peak sales month: {month_names[peak_month-1]} - Maximize inventory and staffing")
            st.write(f"- Lowest sales month: {month_names[low_month-1]} - Focus on promotions and cost control")
            st.write("- Plan inventory cycles based on seasonal patterns")

        # Advanced insights if analyzer is available
        if ADVANCED_ANALYSIS_AVAILABLE and analyzer:
            st.subheader("🔬 Advanced Analytics Results")
            with st.spinner("Running advanced analysis..."):
                try:
                    advanced_results = analyzer.run_complete_advanced_analysis()
                    if advanced_results:
                        st.success("✅ Advanced analysis completed successfully!")
                        
                        # Display feature importance if available
                        if 'feature_importance' in advanced_results:
                            st.write("**📊 Key Sales Drivers:**")
                            for feature, importance in advanced_results['feature_importance'].items():
                                st.write(f"- {feature}: {importance:.3f}")
                        
                        # Display model performance if available
                        if 'model_performance' in advanced_results:
                            st.write("**🎯 Model Accuracy:**")
                            for model, score in advanced_results['model_performance'].items():
                                st.write(f"- {model}: {score:.3f}")
                    else:
                        st.warning("Advanced analysis returned no results.")
                except Exception as e:
                    st.error(f"Advanced analysis failed: {str(e)}")
        else:
            st.info("💡 **Enable Advanced Analytics** by ensuring all required modules are properly installed and configured for deeper insights.")

        st.subheader("🚀 Action Items")
        st.markdown("""
        **Immediate Actions (Next 30 days):**
        - Review underperforming stores and identify root causes
        - Optimize inventory for upcoming seasonal peaks
        - Implement data quality improvements if missing values detected

        **Medium-term Strategy (3-6 months):**
        - Develop store-specific marketing strategies based on performance patterns
        - Enhance forecasting accuracy with additional external data sources
        - Create automated alerts for unusual sales patterns

        **Long-term Vision (6+ months):**
        - Implement predictive analytics for proactive decision making
        - Develop customer segmentation strategies
        - Expand high-performing store formats to new locations
        """)

    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")

if __name__ == "__main__":
    main()