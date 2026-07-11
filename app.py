# app.py
# Step 2: Backend server - receives an image, returns a digit prediction

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import base64
import io

app = Flask(__name__)
CORS(app)  # allows the frontend (different origin) to talk to this backend

# ---------------------------------------------------
# Load the trained model ONCE when the server starts
# (not on every request - that would be slow)
# ---------------------------------------------------
model = load_model("digit_model.h5")


def preprocess_image(image_data_url):
    """
    Takes a base64-encoded image (sent from the browser canvas),
    converts it into the 28x28 grayscale format the model expects.
    """
    # The frontend sends something like "data:image/png;base64,iVBORw0KG..."
    header, encoded = image_data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)

    # Open as a PIL image, convert to grayscale
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    # Resize to 28x28 - same size as MNIST training images
    image = image.resize((28, 28))

    # Convert to numpy array and normalize (same as training preprocessing!)
    img_array = np.array(image).astype("float32") / 255.0

    # MNIST digits are white-on-black; canvas drawings are often black-on-white
    # Invert colors if needed so it matches the training data format
    img_array = 1 - img_array

    # Reshape to match model's expected input: (1, 28, 28, 1)
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_data_url = data["image"]  # base64 string from frontend

    # Preprocess
    processed = preprocess_image(image_data_url)

    # Run prediction
    predictions = model.predict(processed)[0]  # array of 10 probabilities
    predicted_digit = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    return jsonify({
        "digit": predicted_digit,
        "confidence": round(confidence * 100, 2)
    })


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "Backend is running"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
