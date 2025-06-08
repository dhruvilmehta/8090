import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
import xgboost as xgb
import joblib

try:
    with open('public_cases.json', 'r') as file:
        data = json.load(file)
except FileNotFoundError:
    print("Error: 'public_cases.json' not found.")
    exit()

try:
    inputs = []
    outputs = []
    for record in data:
        trip_days = record['input']['trip_duration_days']
        miles = record['input']['miles_traveled']
        receipts = record['input']['total_receipts_amount']
        features = [
            trip_days,
            miles,
            receipts,
            np.log1p(miles * receipts),
            trip_days * receipts
        ]
        inputs.append(features)
        outputs.append(record['expected_output'])
    X = np.array(inputs)
    y = np.log1p(np.array(outputs))
except KeyError:
    print("Error: JSON structure doesn't match expected format.")
    exit()

df = pd.DataFrame(X, columns=['trip_days', 'miles', 'receipts', 'log_miles_times_receipts', 'trip_days_times_receipts'])
df['output'] = np.expm1(y)
print("Data Statistics:")
print(df.describe())
print("\nSkewness:")
print(df.skew())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

z_scores_train = np.abs((y_train - np.mean(y_train)) / np.std(y_train))
non_outlier_mask_train = z_scores_train < 4.0
X_train_clean = X_train[non_outlier_mask_train]
y_train_clean = y_train[non_outlier_mask_train]
print(f"\nOriginal training samples: {len(y_train)}, After outlier removal: {len(y_train_clean)}")

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_clean)
X_test_scaled = scaler.transform(X_test)

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
param_grid_xgb = {
    'n_estimators': [100, 200, 300, 400, 500, 600, 700],
    'max_depth': [3, 4],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'min_child_weight': [1, 3],
    'reg_lambda': [0.1, 1.0]
}
grid_search_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_xgb.fit(X_train_scaled, y_train_clean)
best_xgb = grid_search_xgb.best_estimator_
y_pred_xgb = np.expm1(best_xgb.predict(X_test_scaled))
y_test_orig = np.expm1(y_test)
mse_xgb = mean_squared_error(y_test_orig, y_pred_xgb)
print(f"\nXGBoost MSE: {mse_xgb:.4f}")
print(f"Best XGB Parameters: {grid_search_xgb.best_params_}")

rf = RandomForestRegressor(random_state=42, max_depth=20, min_samples_split=5, min_samples_leaf=2, n_estimators=500)
rf.fit(X_train_scaled, y_train_clean)
y_pred_rf = np.expm1(rf.predict(X_test_scaled))
mse_rf = mean_squared_error(y_test_orig, y_pred_rf)
print(f"Random Forest MSE: {mse_rf:.4f}")

stack_X_train = np.column_stack((rf.predict(X_train_scaled), best_xgb.predict(X_train_scaled)))
stack_X_test = np.column_stack((rf.predict(X_test_scaled), best_xgb.predict(X_test_scaled)))
y_pred_stack = np.expm1(0.8 * best_xgb.predict(X_test_scaled) + 0.2 * rf.predict(X_test_scaled))
mse_stack = mean_squared_error(y_test_orig, y_pred_stack)
print(f"Weighted Stacking Ensemble MSE: {mse_stack:.4f}")

feature_names = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'log_miles_times_receipts', 'trip_days_times_receipts']
importances = rf.feature_importances_
print("\nFeature Importances (Random Forest):")
for name, imp in zip(feature_names, importances):
    print(f"{name}: {imp:.4f}")

sample_input = np.array([[1, 344.46, 813.85, np.log1p(344.46 * 813.85), 1 * 813.85]])
sample_input_scaled = scaler.transform(sample_input)
xgb_pred = np.expm1(best_xgb.predict(sample_input_scaled))
rf_pred = np.expm1(rf.predict(sample_input_scaled))
stack_pred = np.expm1(0.8 * best_xgb.predict(sample_input_scaled) + 0.2 * rf.predict(sample_input_scaled))
print(f"\nXGBoost Predicted output: {xgb_pred[0]:.4f}")
print(f"Random Forest Predicted output: {rf_pred[0]:.4f}")
print(f"Weighted Stacking Ensemble Predicted output: {stack_pred[0]:.4f}")
print(f"Expected output: 707.88")

joblib.dump(best_xgb, 'final_xgb_model.pkl')
joblib.dump(rf, 'final_rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModels saved as 'final_xgb_model.pkl', 'final_rf_model.pkl', 'stacking_model.pkl', and 'scaler.pkl'")
