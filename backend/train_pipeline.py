import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping


def load_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['code', 'date'])
    df = df.fillna(method='ffill')
    return df


def engineer_features(df):
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter

    for lag in range(1, 6):
        df[f'close_lag_{lag}'] = df.groupby('code')['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df.groupby('code')['volume'].shift(lag)

    df['price_change'] = df['close'] - df['open']
    df['price_change_pct'] = (df['close'] - df['open']) / df['open']
    df['daily_range'] = df['high'] - df['low']
    df['daily_range_pct'] = df['daily_range'] / df['open']

    df['sma_cross'] = np.where(df['SMA_5'] > df['SMA_10'], 1, 0)
    df['ema_cross'] = np.where(df['EMA_5'] > df['EMA_10'], 1, 0)
    df['volume_sma5'] = df.groupby('code')['volume'].rolling(window=5).mean().reset_index(0, drop=True)
    df['volume_change'] = df.groupby('code')['volume'].pct_change()
    df['macd_rsi_signal'] = df['MACD'] * df['RSI'] / 100

    df = df.dropna()
    return df


def prepare_data(df, target='close'):
    label_encoder = LabelEncoder()
    df['code'] = label_encoder.fit_transform(df['code'])

    features = [col for col in df.columns if col not in ['date', target]]
    X = df[features]
    y = df[[target]]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    return X_scaled, y_scaled, scaler_X, scaler_y, label_encoder, features


def train_xgboost(X, y):
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=300
    )
    model.fit(X, y.ravel())
    return model


def train_bilstm(X, y, input_dim):
    X_reshaped = X.reshape((X.shape[0], 1, input_dim))

    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(1, input_dim)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X_reshaped, y, epochs=100, batch_size=32, verbose=0, callbacks=[early_stop])

    return model, X_reshaped


def evaluate(model, X, y, scaler_y, is_lstm=False, X_reshaped=None):
    if is_lstm:
        y_pred = model.predict(X_reshaped)
    else:
        y_pred = model.predict(X)

    y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
    y_true_inv = scaler_y.inverse_transform(y)

    rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
    return rmse


def main():
    df = load_data('backend/data/processed_data.csv')
    df = engineer_features(df)
    X, y, scaler_X, scaler_y, label_encoder, features = prepare_data(df)

    model_xgb = train_xgboost(X, y)
    rmse_xgb = evaluate(model_xgb, X, y, scaler_y)

    model_lstm, X_reshaped = train_bilstm(X, y, X.shape[1])
    rmse_lstm = evaluate(model_lstm, X, y, scaler_y, is_lstm=True, X_reshaped=X_reshaped)

    if rmse_xgb < rmse_lstm:
        joblib.dump(model_xgb, 'backend/models/best_model_xgb.pkl')
        print("Saved: best_model_xgb.pkl")
    else:
        model_lstm.save('backend/models/best_model_bilstm.h5')
        print("Saved: best_model_bilstm.h5")

    joblib.dump(scaler_X, 'backend/models/scaler_X.pkl')
    joblib.dump(scaler_y, 'backend/models/scaler_y.pkl')
    joblib.dump(label_encoder, 'backend/models/label_encoder_code.pkl')
    print("Saved scalers and label encoder")

if __name__ == '__main__':
    main()
