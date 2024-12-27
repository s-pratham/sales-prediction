from flask import Flask, request, jsonify 
from flask_cors import CORS  # Import CORS to handle cross-origin requests
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Enable CORS to allow requests from React frontend
CORS(app)

# Load data and preprocess
data = pd.read_csv('train.csv')  # Replace with your dataset's path
data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y')

# Initialize scaler
scaler = MinMaxScaler()
data['sales'] = scaler.fit_transform(data[['sales']])

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
    item_data = data[data['item'] == item].sort_values('date')
    sales = item_data['sales'].values[-sequence_length:]
    monthly_prediction = []
    current_input = sales

    # Reshape to (1, sequence_length, 1) to match LSTM input format
    current_input = np.reshape(current_input, (1, sequence_length, 1))

    for _ in range(days_to_predict):
        next_day = model.predict(current_input)[0][0]
        monthly_prediction.append(next_day)

        # Reshape next_day to 3D before appending
        next_day_reshaped = np.reshape(next_day, (1, 1, 1))

        # Shift window and append next_day_reshaped
        current_input = np.append(current_input[:, 1:, :], next_day_reshaped, axis=1)

    # Inverse transform the predictions
    monthly_prediction = scaler.inverse_transform(np.array(monthly_prediction).reshape(-1, 1))
    total_monthly_sales = monthly_prediction.sum()
    return total_monthly_sales

# Load trained model using TensorFlow's Keras API
def load_trained_model(model_path):
    # Explicitly load model with the correct loss function
    custom_objects = {
        'mse': tf.keras.losses.MeanSquaredError()  # Provide the MeanSquaredError function explicitly
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model

# Load trained model
model = load_trained_model('bilstm_model.h5')  # Save and load your trained model

# API route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        req_data = request.get_json()
        item = int(req_data['item'])  # Get item ID from request
        sequence_length = 30
        days_to_predict = 30

        total_sales = predict_monthly_sales(
            model, data, item, sequence_length=sequence_length, days_to_predict=days_to_predict
        )

        response = {
            'total_sales': float(total_sales)
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
