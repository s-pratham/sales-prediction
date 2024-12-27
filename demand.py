from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load data and preprocess
try:
    data = pd.read_csv('train.csv', low_memory=False, dtype={'item': 'Int64'})  # Support NA values in integers
    data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y')
    data['sales'] = data['sales'].fillna(0)  # Replace missing sales with 0

    # Drop rows with missing item or date
    data = data.dropna(subset=['item', 'date'])

    # Initialize scaler
    scaler = MinMaxScaler()
    data['sales'] = scaler.fit_transform(data[['sales']])
except Exception as e:
    print(f"Error loading or preprocessing data: {e}")
    exit(1)

# Function to prepare sequences
def create_sequences(dataset, sequence_length=30):
    sequences = []
    labels = []
    for i in range(sequence_length, len(dataset)):
        sequences.append(dataset[i-sequence_length:i])
        labels.append(dataset[i])
    return np.array(sequences).reshape(-1, sequence_length, 1), np.array(labels)

# Predict next month's sales
def predict_monthly_sales(model, data, item, sequence_length=30, days_to_predict=30):
    try:
        item_data = data[data['item'] == item].sort_values('date')
        if item_data.empty:
            raise ValueError(f"No data found for item ID: {item}")
        
        sales = item_data['sales'].values[-sequence_length:]
        monthly_prediction = []
        current_input = sales

        current_input = np.reshape(current_input, (1, sequence_length, 1))

        for _ in range(days_to_predict):
            next_day = model.predict(current_input)[0][0]
            monthly_prediction.append(next_day)
            next_day_reshaped = np.reshape(next_day, (1, 1, 1))
            current_input = np.append(current_input[:, 1:, :], next_day_reshaped, axis=1)

        monthly_prediction = scaler.inverse_transform(np.array(monthly_prediction).reshape(-1, 1))
        total_monthly_sales = monthly_prediction.sum()
        return total_monthly_sales
    except Exception as e:
        raise ValueError(f"Error predicting monthly sales: {e}")

# Load trained model
def load_trained_model(model_path):
    try:
        custom_objects = {'mse': tf.keras.losses.MeanSquaredError()}
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

model = load_trained_model('bilstm_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        req_data = request.get_json()
        item = int(req_data['item'])
        sequence_length = 30
        days_to_predict = 30

        total_sales = predict_monthly_sales(
            model, data, item, sequence_length=sequence_length, days_to_predict=days_to_predict
        )

        response = {'total_sales': float(total_sales)}
        return jsonify(response)
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': f"Unexpected error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
