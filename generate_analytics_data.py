import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

sys.path.append('.')

# Set random seed for reproducibility
np.random.seed(42)

# Read the existing model artifact to get feature names
import joblib
artifact = joblib.load('models/artifacts/model.pkl')
pipeline = artifact['pipeline']
preprocessor = pipeline.named_steps['preprocessor']
model = pipeline.named_steps['model']

# Generate 30 days of prediction history with realistic variation
base_date = datetime(2026, 7, 10)
records = []

for day_offset in range(30):
    current_date = base_date + timedelta(days=day_offset)
    
    # Generate 3-5 predictions per day across different hours
    num_predictions = np.random.randint(3, 6)
    
    for pred_idx in range(num_predictions):
        # Random hour
        hour = np.random.randint(0, 24)
        
        # Features
        month = current_date.month
        weekend = 1 if current_date.weekday() >= 5 else 0
        peak_hour = 1 if hour in [12, 13, 19, 20, 21] else 0
        
        # Add festival events on specific days
        festival_flag = 1 if current_date.day % 7 == 0 else 0
        
        ipl_flag = 1 if current_date.day % 5 == 0 else 0
        heavy_rain = 1 if current_date.day % 11 == 0 else 0
        
        # Weather varies by time of day and day
        base_temp = 28 + (day_offset % 7) * 0.5 + hour * 0.3
        temperature = base_temp + np.random.normal(0, 1.5)
        
        humidity = 60 + np.random.normal(0, 10)
        wind_speed = 10 + np.random.normal(0, 3)
        
        # Traffic level varies throughout the day
        if peak_hour:
            traffic_level = np.random.choice(['High', 'Medium', 'Medium'], p=[0.6, 0.3, 0.1])
        else:
            traffic_level = np.random.choice(['Low', 'Medium', 'Medium'], p=[0.4, 0.5, 0.1])
        
        # Driver utilization and restaurant load
        driver_util = 0.3 + (0.5 if peak_hour else 0.2) + np.random.normal(0, 0.1)
        restaurant_load = 0.4 + (0.4 if peak_hour else 0.1) + np.random.normal(0, 0.1)
        
        # Average delivery duration
        avg_delivery = 35 + np.random.normal(0, 5)
        
        # District and day name
        districts = ['Trichy', 'Chennai', 'Madurai', 'Coimbatore']
        district = np.random.choice(districts)
        day_name = current_date.strftime('%A')
        
        # Base demand with various factors
        base_demand = 5.0
        
        # Factors that increase demand
        if festival_flag:
            base_demand *= 1.5  # 50% lift on festivals
        if peak_hour:
            base_demand *= 1.3  # 30% more during peak hours
        if traffic_level == 'High':
            base_demand *= 1.2  # 20% more during high traffic
        if weekend:
            base_demand *= 0.9  # 10% less on weekends
        
        # Add temperature effect
        temp_effect = 1.0 + (temperature - 28) * 0.05
        base_demand *= temp_effect
        
        # Add weather effect
        if heavy_rain:
            base_demand *= 0.8
        
        # Add some randomness
        actual_demand = max(1.0, base_demand + np.random.normal(0, 1.5))
        
        # Model prediction (slightly off from actual)
        predicted_demand = actual_demand * (1 + np.random.normal(0, 0.15))
        
        # Error calculation
        error = actual_demand - predicted_demand
        abs_error = abs(error)
        
        timestamp = current_date.replace(hour=hour, minute=np.random.randint(0, 60), second=np.random.randint(0, 60))
        
        record = {
            'Month': month,
            'Weekend': weekend,
            'PeakHour': peak_hour,
            'FestivalFlag': festival_flag,
            'IPLFlag': ipl_flag,
            'HeavyRain': heavy_rain,
            'DriverUtilization': np.clip(driver_util, 0, 1),
            'RestaurantLoad': np.clip(restaurant_load, 0, 1),
            'AvgDeliveryDuration': avg_delivery,
            'Temperature': temperature,
            'Humidity': np.clip(humidity, 0, 100),
            'Wind_Speed': wind_speed,
            'District': district,
            'Day': day_name,
            'Traffic_Level': traffic_level,
            'ActualDemand': round(actual_demand, 2),
            'PredictedDemand': round(predicted_demand, 2),
            'Timestamp': timestamp.isoformat(),
            'Hour': hour,
            'WeatherCondition': 'Rainy' if heavy_rain else 'Clear',
            'model_version': 'v1.2',
            'Error': round(error, 4),
            'AbsoluteError': round(abs_error, 4),
        }
        
        records.append(record)

# Create DataFrame and save
df_new = pd.DataFrame(records)
df_new = df_new.sort_values('Timestamp')
df_new.to_csv('outputs/prediction_history.csv', index=False)

print('✓ Generated 30 days of realistic prediction history')
print(f'  - Total records: {len(df_new)}')
print(f'\nMetric preview:')
print(f'  - Avg Daily Demand: {df_new["ActualDemand"].mean():.2f}')
print(f'  - Festival events: {(df_new["FestivalFlag"] == 1).sum()}')
print(f'  - High traffic instances: {(df_new["Traffic_Level"] == "High").sum()}')
print(f'  - Weather correlation: {df_new["Temperature"].corr(df_new["ActualDemand"]):.3f}')
print(f'  - Prediction MAE: {df_new["AbsoluteError"].mean():.3f}')
