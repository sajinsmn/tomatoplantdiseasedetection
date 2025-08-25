import os
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg to avoid GUI issues
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import gdown  # to download from Google Drive

# Initialize Flask app
app = Flask(__name__)

# Setup upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Path to model
MODEL_PATH = "inceptionv3_model.h5"

# 🔹 Google Drive direct download link (replace with your link)
MODEL_URL = "https://drive.google.com/uc?export=download&id=1mCsJrj-omqKBoOjcPObCGiIZLfGV3GMe"

# Check if model exists, otherwise download
if not os.path.exists(MODEL_PATH):
    print("Model not found. Downloading from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load the trained model
model = load_model(MODEL_PATH)

# Class labels as per the model's training
class_names = ['Early blight', 'Leaf miner', 'Mealy bug', 'Healthy']

# Pesticide recommendations
pesticide_recommendations = {
    'Early blight': 'Chlorothalonil, Copper, Azoxystrobin, or Difenoconazole.',
    'Leaf miner': 'Spinosad, Abamectin, Chlorantraniliprole, or Spinetoram.',
    'Mealy bug': 'Dichlorvos, Chlorpyriphos, Azadirachtin, or Buprofezin.',
    'Healthy': 'No pesticide needed.'
}

# Preprocess image for prediction
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Resize
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Save uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Predict
    img_array = preprocess_image(file_path)
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class = class_names[predicted_class_idx]
    recommendation = pesticide_recommendations[predicted_class]

    # Plot uploaded image
    img = image.load_img(file_path)
    plt.imshow(img)
    plt.axis('off')
    plt.savefig('static/result.png')

    # Remove uploaded file
    os.remove(file_path)

    # Convert image to base64
    img_io = BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    img_b64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

    # Render result
    return render_template(
        'result.html',
        predicted_class=predicted_class,
        img_b64=img_b64,
        recommendation=recommendation
    )

# Run Flask
if __name__ == '__main__':
    app.run(debug=True)
