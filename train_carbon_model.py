# train_carbon_model.py

import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# 1. Load your dataset
df = pd.read_csv('Carbon_Emission.csv')  # Update filename if needed

# 2. Handle missing values (drop or fill as appropriate)
df = df.dropna()

# 3. Expand list-like columns into binary columns (e.g., Recycling, Cooking_With)
list_like_columns = ['Recycling', 'Cooking_With']
for col in list_like_columns:
    if col in df.columns:
        # Convert string representations of lists to actual lists
        df[col] = df[col].fillna('[]').apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
        # Find all unique items
        unique_items = set()
        for items in df[col]:
            unique_items.update(items)
        # Create binary columns for each unique item
        for item in unique_items:
            df[f'{col}_{item}'] = df[col].apply(lambda x: item in x)
        df.drop(columns=[col], inplace=True)

# 4. Identify and encode all remaining categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
if 'CarbonEmission' in categorical_cols:
    categorical_cols.remove('CarbonEmission')

df_encoded = pd.get_dummies(df, columns=categorical_cols)

# 5. Ensure all columns are numeric (convert bools to int)
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'bool':
        df_encoded[col] = df_encoded[col].astype(int)

# 6. Separate features and target
X = df_encoded.drop('CarbonEmission', axis=1)
y = df_encoded['CarbonEmission']

# 7. Check for any non-numeric columns
non_numeric = [col for col in X.columns if not np.issubdtype(X[col].dtype, np.number)]
if non_numeric:
    print("Non-numeric columns still present:", non_numeric)
    raise ValueError("All features must be numeric. Please check your dataset.")

# 8. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 9. Train the RandomForestRegressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 10. Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"  MAE: {mae:.2f}")
print(f"  MSE: {mse:.2f}")
print(f"  R²: {r2:.2f}")

# 11. Save the trained model and feature columns
joblib.dump(model, 'carbon_model.pkl')
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')
print("Model and feature columns saved as 'carbon_model.pkl' and 'feature_columns.pkl'.")

# 12. (Optional) Show feature importance
importances = model.feature_importances_
features = X.columns
indices = importances.argsort()[::-1]

plt.figure(figsize=(10,6))
plt.title("Feature Importances")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), features[indices], rotation=90)
plt.tight_layout()
plt.show()
