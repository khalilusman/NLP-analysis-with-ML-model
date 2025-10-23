"""
Quick fix - Clean the saved column names to match what model actually uses
Run this in your backend folder
"""

import pickle

# Load existing columns
with open('../saved_models/train_columns.pkl', 'rb') as f:
    original_columns = pickle.load(f)

print(f"Original columns: {len(original_columns)}")
print(f"Sample original: {original_columns[:3]}")

# Clean them the same way training did
import pandas as pd
cleaned_columns = pd.Series(original_columns).str.replace('[^A-Za-z0-9_]+', '_', regex=True).tolist()

print(f"\nCleaned columns: {len(cleaned_columns)}")
print(f"Sample cleaned: {cleaned_columns[:3]}")

# Save cleaned version
with open('../saved_models/train_columns.pkl', 'wb') as f:
    pickle.dump(cleaned_columns, f)

print("\n Columns fixed and saved!")
print("Restart your backend now.")