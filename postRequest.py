from predict import predictImage
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins for all routes

@app.route('/predictImage', methods=['POST'])
def predict_label():
    if 'image' in request.files:
        image = request.files['image']
        image.save('./uploaded_image.jpeg')
        image = cv2.imread('./uploaded_image.jpeg')
        predicted_label = predictImage(image)
        return jsonify({'predicted_label': predicted_label})
    else:
        return 'No image uploaded.'            


if __name__ == '__main__':
    app.run(debug=True)