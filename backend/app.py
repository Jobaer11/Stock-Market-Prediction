from flask import Flask, request, jsonify
from model_loader import load_all
from utils import prepare_features_and_predict

app = Flask(__name__)
model, scaler_X, scaler_y, label_encoder, df = load_all()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    company_name = data.get('company')
    date = data.get('date')

    try:
        code = label_encoder.transform([company_name])[0]
        prediction = prepare_features_and_predict(df, model, code, date, scaler_X, scaler_y)
        return jsonify({'prediction': float(round(prediction, 2))})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/companies', methods=['GET'])
def companies():
    return jsonify({'companies': list(label_encoder.classes_)})

if __name__ == '__main__':
    app.run(debug=True)
