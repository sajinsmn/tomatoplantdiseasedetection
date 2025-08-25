import os
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import gdown

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_PATH = "inceptionv3_model.h5"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1mCsJrj-omqKBoOjcPObCGiIZLfGV3GMe"

if not os.path.exists(MODEL_PATH):
    print("Model not found. Downloading from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)

class_names = ['Early blight', 'Leaf miner', 'Mealy bug', 'Healthy']

pesticide_recommendations = {
    'Early blight': 'Chlorothalonil, Copper, Azoxystrobin, or Difenoconazole.',
    'Leaf miner': 'Spinosad, Abamectin, Chlorantraniliprole, or Spinetoram.',
    'Mealy bug': 'Dichlorvos, Chlorpyriphos, Azadirachtin, or Buprofezin.',
    'Healthy': 'No pesticide needed.'
}

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    img_array = preprocess_image(file_path)
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class = class_names[predicted_class_idx]
    recommendation = pesticide_recommendations[predicted_class]

    img = image.load_img(file_path)
    plt.imshow(img)
    plt.axis('off')
    plt.savefig('static/result.png')

    os.remove(file_path)

    img_io = BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    img_b64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

    return render_template(
        'result.html',
        predicted_class=predicted_class,
        img_b64=img_b64,
        recommendation=recommendation
    )

if __name__ == '__main__':
    app.run(debug=True)
