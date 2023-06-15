FROM python:3.8

# Install any necessary dependencies specific to your API
RUN pip install flask joblib scikit-learn pandas

# Create the necessary directories
RUN mkdir -p /app/MODEL_DIR
RUN mkdir -p /app/api

# Copy the required files into the image
COPY train.csv /app/train.csv
COPY test.csv /app/test.csv
COPY train.py /app/train.py
COPY inference.py /app/inference.py
COPY MODEL_DIR/clf_lda.joblib /app/MODEL_DIR/clf_lda.joblib
COPY MODEL_DIR/clf_nn.joblib /app/MODEL_DIR/clf_nn.joblib
COPY api/api.py /app/api/api.py

# Set the working directory
WORKDIR /app

# Expose the necessary port(s) for your API
EXPOSE 80

# Start the API
CMD ["python", "-m", "api.api"]
