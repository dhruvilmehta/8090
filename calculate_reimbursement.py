import sys
import numpy as np
import joblib

def main():
    # print("Called")
    if len(sys.argv) != 4:
        print("Usage: python3 calculate_reimbursement.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1)

    # Parse input arguments
    trip_duration_days = float(sys.argv[1])
    miles_traveled = float(sys.argv[2])
    total_receipts_amount = float(sys.argv[3])
    log_receipts = np.log1p(miles_traveled * total_receipts_amount)  # if you used log1p in training

    # Load scaler and models
    scaler = joblib.load('scaler.pkl')
    xgb_model = joblib.load('final_xgb_model.pkl')
    rf_model = joblib.load('final_rf_model.pkl')

    # Prepare input features (match the training order)
    sample_input = np.array([[trip_duration_days, miles_traveled, total_receipts_amount, log_receipts, trip_duration_days*total_receipts_amount]])

    # Scale input features
    sample_input_scaled = scaler.transform(sample_input)

    # Predict with each model
    xgb_pred = xgb_model.predict(sample_input_scaled)
    rf_pred = rf_model.predict(sample_input_scaled)

    # Weighted ensemble prediction — adjust weight_rf as you want
    weight_rf = 0.3  # example weight for RF
    ensemble_pred = weight_rf * rf_pred + (1 - weight_rf) * xgb_pred

    # Print final prediction
    print(f"{np.expm1(ensemble_pred[0]):.4f}")

if __name__ == "__main__":
    main()
