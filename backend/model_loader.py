import joblib
import pandas as pd
from tensorflow.keras.models import load_model

def load_all():
    model = joblib.load('models/best_model_xgb.pkl')  # or load_model('models/best_model_bilstm.h5')
    scaler_X = joblib.load('models/scaler_X.pkl')
    scaler_y = joblib.load('models/scaler_y.pkl')
    label_encoder = joblib.load('models/label_encoder_code.pkl')
    df = pd.read_csv('data/processed_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    return model, scaler_X, scaler_y, label_encoder, df
