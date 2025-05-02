import os
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)  # Allow all origins (restrict in production)

# Global variables for model, scaler, and PCA (loaded lazily)
model = None
scaler = None
pca = None

def load_models():
    global model, scaler, pca
    if model is None or scaler is None or pca is None:
        print("Loading TensorFlow model, scaler, and PCA...")
        start_time = time.time()
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model('soil_ph_model.h5', compile=False)
            scaler = joblib.load('rgb_scaler.joblib')
            pca = joblib.load('pca_model.joblib')
            print(f"Models loaded successfully in {time.time() - start_time} seconds")
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise

# Optional: Check server status
@app.route('/', methods=['GET'])
def home():
    return "Flask Soil pH Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    print("Incoming data:", request.json)
    start_time = time.time()  # Start timer

    # Load models if not already loaded
    try:
        load_models()
    except Exception as e:
        print(f"Model loading failed: {str(e)}, took {time.time() - start_time} seconds")
        return jsonify({'error': 'Failed to load models'}), 500

    try:
        # Validate JSON input
        if not request.json or 'rgb' not in request.json:
            print(f"Prediction failed: Missing 'rgb' key in JSON, took {time.time() - start_time} seconds")
            return jsonify({'error': 'Missing "rgb" key in JSON'}), 400
        data = request.json['rgb']
        if not isinstance(data, list) or not all(isinstance(x, (int, float)) for x in data):
            print(f"Prediction failed: 'rgb' must be a list of numbers, took {time.time() - start_time} seconds")
            return jsonify({'error': '"rgb" must be a list of numbers'}), 400
        
        # Process prediction
        rgb_values = np.array([data])
        rgb_normalized = scaler.transform(rgb_values)
        pca_components = pca.transform(rgb_normalized)
        ph_pred = model.predict(pca_components, verbose=0)[0][0]
        print(f"Prediction successful, took {time.time() - start_time} seconds")
        return jsonify({'ph': float(ph_pred)})
    except Exception as e:
        print(f"Prediction error: {str(e)}, took {time.time() - start_time} seconds")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use PORT from Render, default to 5000 locally
    print("Flask Soil pH Prediction API is starting...")
    app.run(host='0.0.0.0', port=port)