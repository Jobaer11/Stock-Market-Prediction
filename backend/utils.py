from train_pipeline import engineer_features
import pandas as pd
import numpy as np

def prepare_features_and_predict(df_full, model, company_code, target_date, scaler_X, scaler_y, n_days=60):
    target_date = pd.to_datetime(target_date)
    df_c = df_full[df_full['code'] == company_code].copy()
    df_c = df_c[df_c['date'] < target_date].copy()

    # Calculate closeness to target date (only for filtering, not used in model)
    df_c['month_diff'] = abs(df_c['date'].dt.month - target_date.month)
    df_c['day_diff'] = abs(df_c['date'].dt.day - target_date.day)

    # Sort to prioritize rows closer in time to target date (by month and day only)
    df_c = df_c.sort_values(['month_diff', 'day_diff', 'date'])
    df_c = df_c.drop(columns=['month_diff', 'day_diff'])

    # Take the most relevant n_days for contextual feature calculation
    df_c = df_c.head(n_days).sort_values('date')

    if len(df_c) < n_days:
        raise ValueError("Not enough relevant historical data for feature computation")

    # Engineer features
    df_c_fe = engineer_features(df_c.copy())

    # Use the last row and override date-based fields
    prediction_row = df_c_fe.iloc[-1:].copy()
    prediction_row['date'] = target_date
    prediction_row['day_of_week'] = target_date.dayofweek
    prediction_row['month'] = target_date.month
    prediction_row['quarter'] = target_date.quarter

    # Ensure feature alignment with model input
    X_input = scaler_X.transform(prediction_row.drop(columns=['date', 'close']))
    y_pred = model.predict(X_input)
    return scaler_y.inverse_transform(y_pred.reshape(-1, 1))[0][0]
