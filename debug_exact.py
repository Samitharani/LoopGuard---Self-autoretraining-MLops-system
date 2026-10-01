import sys, pandas as pd, numpy as np, json
sys.path.append('.')

PREDICTION_HISTORY_CSV = 'outputs/prediction_history.csv'

history = pd.read_csv(PREDICTION_HISTORY_CSV)

# Calculate Error and AbsoluteError if missing
if "Error" not in history.columns or history["Error"].isna().all():
    history["Error"] = history["ActualDemand"] - history["PredictedDemand"]
if "AbsoluteError" not in history.columns or history["AbsoluteError"].isna().all():
    history["AbsoluteError"] = (history["ActualDemand"] - history["PredictedDemand"]).abs()

# Extract Hour from Timestamp if missing
if "Hour" not in history.columns or history["Hour"].isna().sum() > len(history) * 0.5:
    history["Hour"] = pd.to_datetime(history["Timestamp"], errors="coerce").dt.hour

print("Before replace/where:")
print("Column dtypes:", history.dtypes)

# Try the replace and where operations
history = history.replace({pd.NA: None, float('inf'): None, float('-inf'): None})
print("After replace, any NaN left?", history.isna().sum().sum())

history = history.where(pd.notnull(history), None)
print("After where, any NaN left?", history.isna().sum().sum())

# Try to convert to dict
print("\nTrying to_dict:")
records = history.iloc[0:3].to_dict(orient="records")
print("Got records, trying JSON serialization:")

try:
    json_str = json.dumps(records)
    print("SUCCESS!")
except ValueError as e:
    print(f"FAILED: {e}")
    # Find which record/field has the issue
    for i, rec in enumerate(records):
        for k, v in rec.items():
            if isinstance(v, float):
                if np.isnan(v) or np.isinf(v):
                    print(f"Record {i}, field {k}: value={v}, isnan={np.isnan(v)}, isinf={np.isinf(v)}")
