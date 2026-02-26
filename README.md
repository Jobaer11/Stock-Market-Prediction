# Stock Market Prediction

A machine learning-based web application for predicting stock market trends using advanced forecasting models.

## Project Overview

This project combines machine learning models (XGBoost, TensorFlow) with a user-friendly web interface to predict stock market movements. It includes both a Flask backend API and a Streamlit frontend for easy interaction.

## Project Structure

```
stock_market_prediction/
├── backend/                    # Flask API backend
│   ├── app.py                 # Main Flask application
│   ├── model_loader.py        # Model loading utilities
│   ├── train_pipeline.py      # Model training pipeline
│   ├── utils.py               # Utility functions
│   ├── data/
│   │   └── processed_data.csv # Processed dataset
│   └── models/                # Trained model storage
├── frontend/
│   └── streamlit_app.py       # Streamlit web interface
├── requirements.txt           # Project dependencies
├── test_tf.py                # TensorFlow tests
└── env/                       # Virtual environment
```

## Features

- Stock data preprocessing and feature engineering
- Multiple ML models (XGBoost, TensorFlow) for prediction
- REST API backend with Flask
- Interactive web interface with Streamlit
- Data visualization and analysis tools

## Installation

1. **Clone the repository and navigate to the project directory:**

    ```bash
    cd stock_market_prediction
    ```

2. **Create and activate a virtual environment:**

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Run the Backend API

```bash
cd backend
python app.py
```

The API will be available at `http://localhost:5000`

### Run the Frontend

```bash
streamlit run frontend/streamlit_app.py
```

The web interface will open at `http://localhost:8501`

### Train Models

```bash
cd backend
python train_pipeline.py
```

## Requirements

- Python 3.8+
- Flask
- Streamlit
- scikit-learn
- XGBoost
- TensorFlow
- pandas
- numpy
- matplotlib

See `requirements.txt` for complete dependencies.

## Dataset

The project uses stock market data processed and stored in `data/processed_data.csv`. Input data includes:

- Historical stock prices
- Trading volumes
- Technical indicators
- Market features

## Models

- **XGBoost**: Gradient boosting model for structured prediction
- **TensorFlow**: Deep learning model for time-series forecasting

## License

This project is provided as-is for educational purposes.

## Author

Mehedi

## Contact

For questions or issues, please refer to the project documentation or contact the developer.
