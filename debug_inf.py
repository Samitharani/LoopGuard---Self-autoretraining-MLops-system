import sys, pandas as pd, numpy as np, json
sys.path.append('.')

df = pd.read_csv('outputs/prediction_history.csv')
df['Error'] = df['ActualDemand'] - df['PredictedDemand']
df['AbsoluteError'] = (df['ActualDemand'] - df['PredictedDemand']).abs()
df['Hour'] = pd.to_datetime(df['Timestamp'], errors='coerce').dt.hour

print("Checking for inf values in numeric columns:")
for col in df.select_dtypes(include=[np.number]).columns:
    inf_count = np.isinf(df[col]).sum()
    if inf_count > 0:
        print(f'{col}: {inf_count} inf values')

print("\nTrying to convert to JSON:")
df_clean = df.replace({pd.NA: None, np.inf: None, -np.inf: None})
df_clean = df_clean.where(pd.notnull(df_clean), None)
records = df_clean.head(3).to_dict(orient='records')
try:
    json_str = json.dumps(records)
    print("Success! JSON serialization works")
except Exception as e:
    print(f"Error: {e}")
    print("Sample record:", records[0])
