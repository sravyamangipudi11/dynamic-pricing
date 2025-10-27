import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load model and data
@st.cache_resource
def load_model():
    return joblib.load('xgboost_model.pkl')

@st.cache_data
def load_data():
    personal_data = pd.read_csv("data/My Uber Drives - 2016.csv")
    personal_data.columns = ['START_DATE', 'END_DATE', 'CATEGORY', 'START', 'STOP', 'MILES', 'PURPOSE']
    personal_data = personal_data.dropna()
    personal_data['START'] = personal_data['START'].str.replace('\?', 'a', regex=True)
    personal_data['STOP'] = personal_data['STOP'].str.replace('\?', 'a', regex=True)
    personal_data['START_DATE'] = pd.to_datetime(personal_data['START_DATE'], format="%m/%d/%Y %H:%M")
    personal_data['END_DATE'] = pd.to_datetime(personal_data['END_DATE'], format="%m/%d/%Y %H:%M")
    personal_data['HOUR'] = personal_data['START_DATE'].dt.hour
    personal_data['DAY'] = personal_data['START_DATE'].dt.day
    personal_data['MONTH'] = personal_data['START_DATE'].dt.month
    personal_data['WEEKDAY'] = personal_data['START_DATE'].dt.day_name()
    personal_data['DAY_OF_WEEK'] = personal_data['START_DATE'].dt.dayofweek

    boston_data = pd.read_csv("data/rideshare_kaggle.csv")
    drop_cols = ['apparentTemperature', 'precipIntensity', 'humidity', 'windSpeed', 'apparentTemperatureHigh', 'dewPoint', 
             'precipIntensityMax', 'apparentTemperatureMax', 'cloudCover', 'moonPhase', 'windGustTime', 'visibility', 
             'temperatureHighTime', 'apparentTemperatureHighTime', 'apparentTemperatureLow', 'apparentTemperatureLowTime', 
             'temperatureMinTime', 'temperatureMaxTime', 'apparentTemperatureMin', 'apparentTemperatureMinTime', 
             'apparentTemperatureMaxTime', 'windBearing', 'sunriseTime', 'uvIndex', 'visibility.1', 'ozone', 'sunsetTime', 'uvIndexTime']
    boston_data = boston_data.drop(columns=drop_cols).dropna()
    boston_data = boston_data[boston_data['cab_type'] == 'Uber']
    boston_data['datetime'] = pd.to_datetime(boston_data['datetime'])
    boston_data['WEEKDAY'] = boston_data['datetime'].dt.day_name()
    boston_data['IS_PEAK'] = boston_data['hour'].apply(lambda x: 1 if x in [7, 8, 9, 17, 18, 19] else 0)
    boston_data['IS_HOLIDAY'] = boston_data.apply(lambda x: 1 if x['month'] == 12 and x['day'] in [24, 25, 31] else 0, axis=1)
 
    boston_data['source'] = boston_data['source'].where(boston_data['source'].notna(), None)
    boston_data['destination'] = boston_data['destination'].where(boston_data['destination'].notna(), None)
    boston_data['source'] = boston_data['source'].apply(lambda x: str(x) if pd.notna(x) else "")
    boston_data['destination'] = boston_data['destination'].apply(lambda x: str(x) if pd.notna(x) else "")

    boston_data['IS_AIRPORT'] = (
        boston_data['source'].str.contains('Airport', case=False, na=False) |
        boston_data['destination'].str.contains('Airport', case=False, na=False)
    )
    
    unique_sources = sorted([x for x in boston_data['source'].unique() if x and not x.isdigit()])
    unique_destinations = sorted([x for x in boston_data['destination'].unique() if x and not x.isdigit()])
    unique_names = boston_data['name'].dropna().unique().tolist()


    return personal_data, boston_data, unique_sources, unique_destinations, unique_names

model = load_model()
personal_data, boston_data, unique_sources, unique_destinations, unique_names = load_data()

# Apply background image with CSS
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://ubernewsroomapi.10upcdn.com/wp-content/uploads/sites/374/2015/11/uber_festival-dark-background_blog_700x300-1.png');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .st-expander {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page", ["Home", "EDA", "Price Prediction", "Dynamic Pricing"])

st.sidebar.markdown("""
### About
- **Data**: Personal Uber trips (2016) and Boston Uber rides (Dec 2018).
- **Model**: XGBoost (R²=0.95).
- **Dynamic Pricing**: Increases price by $2.43/ride, $978K/year.
""")

if page == "Home":
    st.title("Uber Ride Price Prediction & Dynamic Pricing App")
    st.markdown("""
    Welcome to the Uber Ride Analysis App! This tool predicts ride prices using XGBoost (R²=0.95) and simulates dynamic pricing based on peak hours, holidays, weather, and airport trips. Key results:
    - Average price increase: **$2.43 per ride**.
    - Estimated annual revenue increase: **$978,247**.
    Explore EDA, predict prices, or simulate dynamic pricing below.
    """)

elif page == "EDA":
    st.title("Exploratory Data Analysis")
    st.subheader("Personal Uber Data")
    fig, ax = plt.subplots()
    sns.countplot(x='CATEGORY', data=personal_data, palette='viridis', ax=ax)
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    sns.countplot(y='PURPOSE', data=personal_data, palette='magma', ax=ax)
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    personal_data['MILES'].plot.hist(bins=20, color='skyblue', ax=ax)
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    personal_data['HOUR'].value_counts().sort_index().plot(kind='bar', color='red', ax=ax)
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    personal_data['WEEKDAY'].value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).plot(kind='barh', color='seagreen', ax=ax)
    st.pyplot(fig)
    
    st.subheader("Boston Uber Data")
    fig, ax = plt.subplots()
    boston_data['hour'].value_counts().sort_index().plot(kind='bar', color='brown', ax=ax)
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    boston_data['WEEKDAY'].value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).plot(kind='barh', color='darkred', ax=ax)
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    sns.countplot(x='IS_HOLIDAY', data=boston_data, palette='viridis', ax=ax)
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    sns.countplot(x='IS_AIRPORT', data=boston_data, palette='magma', ax=ax)
    st.pyplot(fig)

elif page == "Price Prediction":
    st.title("Uber Ride Price Prediction")
    st.markdown("Enter details to predict the base price of an Uber ride. Use dropdowns for intuitive input.")
    
    with st.expander("How to Use", expanded=False):
        st.write("""
        - **Hour**: The time of day (0-23).
        - **Source**: Starting location (e.g., North Station).
        - **Destination**: Ending location (e.g., Airport).
        - **Cab Type**: Uber service type (e.g., UberX).
        - **Distance**: Trip distance in miles.
        The model predicts based on encoded data trained on Boston rides.
        """)
    
    hour = st.slider("Hour of Day (0-23)", 0, 23, 12)
    source = st.selectbox("Source Location", ['Select'] + unique_sources, index=0)
    destination = st.selectbox("Destination Location", ['Select'] + unique_destinations, index=0)
    name = st.selectbox("Cab Type", ['Select'] + unique_names, index=0)
    distance = st.number_input("Distance (miles)", min_value=0.0, value=5.0)
    
    if source != 'Select' and destination != 'Select' and name != 'Select':
        # Encode inputs using LabelEncoder (mimic training data encoding)
        le_source = {val: idx for idx, val in enumerate(sorted(unique_sources))}
        le_destination = {val: idx for idx, val in enumerate(sorted(unique_destinations))}
        le_name = {val: idx for idx, val in enumerate(sorted(unique_names))}
        
        input_data = np.array([[hour, le_source[source], le_destination[destination], le_name[name], distance]])
        prediction = model.predict(input_data)
        st.success(f"Predicted Base Price: ${prediction[0]:.2f}")
    else:
        st.warning("Please select valid Source, Destination, and Cab Type.")

elif page == "Dynamic Pricing":
    st.title("Dynamic Pricing Simulation")
    st.markdown("Simulate an Uber ride price with dynamic surges based on conditions.")
    
    with st.expander("How to Use", expanded=False):
        st.write("""
        - **Hour**: Time of day (0-23) to check peak hours (7-9, 17-19).
        - **Source/Destination**: Locations to check airport trips.
        - **Cab Type**: Service type.
        - **Distance**: Trip distance in miles.
        - **Conditions**: Check boxes for peak hour, holiday, or airport trip; set precipitation probability (0-1).
        Surges: 1.5x (peak), 1.3x (holiday), 1.2x (weather > 0.5), 1.2x (airport).
        """)
    
    hour = st.slider("Hour of Day (0-23)", 0, 23, 12)
    source = st.selectbox("Source Location", ['Select'] + unique_sources, index=0)
    destination = st.selectbox("Destination Location", ['Select'] + unique_destinations, index=0)
    name = st.selectbox("Cab Type", ['Select'] + unique_names, index=0)
    distance = st.number_input("Distance (miles)", min_value=0.0, value=5.0)
    is_peak = st.checkbox("Peak Hour (7-9 AM or 5-7 PM)")
    is_holiday = st.checkbox("Holiday (Dec 24, 25, 31)")
    precip_probability = st.slider("Precipitation Probability (0-1)", 0.0, 1.0, 0.0)
    is_airport = st.checkbox("Airport Trip")
    
    if source != 'Select' and destination != 'Select' and name != 'Select':
        # Encode inputs
        le_source = {val: idx for idx, val in enumerate(sorted(unique_sources))}
        le_destination = {val: idx for idx, val in enumerate(sorted(unique_destinations))}
        le_name = {val: idx for idx, val in enumerate(sorted(unique_names))}
        
        input_data = np.array([[hour, le_source[source], le_destination[destination], le_name[name], distance]])
        base_price = model.predict(input_data)[0]
        
        surge = 1.0
        if is_peak:
            surge *= 1.5
        if is_holiday:
            surge *= 1.3
        if precip_probability > 0.5:
            surge *= 1.2
        if is_airport:
            surge *= 1.2
        
        dynamic_price = base_price * surge
        st.success(f"Base Price: ${base_price:.2f} | Dynamic Price: ${dynamic_price:.2f} | Increase: ${dynamic_price - base_price:.2f}")
    else:
        st.warning("Please select valid Source, Destination, and Cab Type.")
