from flask import Flask, request, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import pickle
from PIL import Image

app = Flask(__name__)

model = load_model("full_trained_model.h5")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

IMG_SIZE = (32, 32)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image = Image.open(file.stream).convert("RGB")
    image = image.resize(IMG_SIZE)
    img = np.array(image).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_index = np.argmax(pred)
    label = le.inverse_transform([class_index])[0]

    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=10000)
