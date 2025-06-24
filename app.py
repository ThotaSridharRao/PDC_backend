import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from utils.predict import load_model_and_predict

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    img_file = request.files['image']
    result = load_model_and_predict(img_file)
    return jsonify({"prediction": result})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))  # Default port is 5000
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
