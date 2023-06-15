import os
import pandas as pd
from joblib import load
from flask import Flask, request, jsonify
import numpy as np

# # Set the paths for the model files
# MODEL_DIR = "/app/MODEL_DIR"
# MODEL_FILE_LDA = os.environ.get("MODEL_FILE_LDA", "clf_lda.joblib")
# MODEL_FILE_NN = os.environ.get("MODEL_FILE_NN", "clf_nn.joblib")
# MODEL_PATH_LDA = os.path.join(MODEL_DIR, MODEL_FILE_LDA)
# MODEL_PATH_NN = os.path.join(MODEL_DIR, MODEL_FILE_NN)
#
# # Load the trained models
# model_lda = load(MODEL_PATH_LDA)
# model_nn = load(MODEL_PATH_NN)

# # Create the Flask application
# app = Flask(__name__)

# Define the API routes
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#
#     # Access the input data from the 'data' key
#     input_data = data['data']
#
#     try:
#         # Reshape the input data to have the correct shape with 160 features
#         reshaped_data = np.array(input_data).reshape(1, 160)
#
#         # Make predictions using the loaded models
#         predictions_lda = model_lda.predict(reshaped_data)
#         predictions_nn = model_nn.predict(reshaped_data)
#
#         # Prepare the response
#         response = {
#             'predictions_lda': predictions_lda.tolist(),
#             'predictions_nn': predictions_nn.tolist()
#         }
#
#         return jsonify(response)
#     except Exception as e:
#         return jsonify({'error': str(e)})

import base64
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/load_image', methods=['POST'])
def load_image():
    # Check if an image file is present in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found'})

    image_file = request.files['image']
    image_data = image_file.read()

    #Decode the BASE 64 - if you get input like
    #image_bytes = base64.b64decode(image_data)
    print(image_data)

    # Create an image object from the decoded bytes
    image = Image.open(BytesIO(image_data))         #(image_bytes)

    # For now, we'll just print a message indicating that the image has been loaded
    print('Image loaded successfully')

    # Return a response indicating that the image has been loaded
    return jsonify({'message': 'Image loaded successfully'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)

