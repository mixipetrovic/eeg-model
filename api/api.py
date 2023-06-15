import os
import pandas as pd
from joblib import load
from flask import Flask, request, jsonify
import numpy as np

# Set the paths for the model files
MODEL_DIR = "/app/MODEL_DIR"
MODEL_FILE_LDA = os.environ.get("MODEL_FILE_LDA", "clf_lda.joblib")
MODEL_FILE_NN = os.environ.get("MODEL_FILE_NN", "clf_nn.joblib")
MODEL_PATH_LDA = os.path.join(MODEL_DIR, MODEL_FILE_LDA)
MODEL_PATH_NN = os.path.join(MODEL_DIR, MODEL_FILE_NN)

# Load the trained models
model_lda = load(MODEL_PATH_LDA)
model_nn = load(MODEL_PATH_NN)

# Create the Flask application
app = Flask(__name__)

# Define the API routes
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Access the input data from the 'data' key
    input_data = data['data']

    try:
        # Reshape the input data to have the correct shape with 160 features
        reshaped_data = np.array(input_data).reshape(1, 160)

        # Make predictions using the loaded models
        predictions_lda = model_lda.predict(reshaped_data)
        predictions_nn = model_nn.predict(reshaped_data)

        # Prepare the response
        response = {
            'predictions_lda': predictions_lda.tolist(),
            'predictions_nn': predictions_nn.tolist()
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)