import os
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Initialize Flask app
app = Flask(__name__)

# Setup upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = load_model('inceptionv3_model.h5')

# Class labels as per the model's training
class_names = ['Early_blight', 'Leaf_miner', 'Mealy_bug', 'Healthy']

# Preprocess image for prediction
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Resize image to the input size expected by the model
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Home route to display the upload form
@app.route('/')
def index():
    return render_template('index.html')

# Predict route to handle image uploads and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400  # Error handling if no file is selected
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400  # Error handling if file name is empty

    # Save the uploaded file to the server
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Preprocess the uploaded image and make prediction
    img_array = preprocess_image(file_path)
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class = class_names[predicted_class_idx]

    # Generate and save the plot of the uploaded image with the predicted class
    img = image.load_img(file_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Predicted: {predicted_class}')
    plt.savefig('static/result.png')  # Save the image with prediction as a static file

    # Remove the uploaded file after processing
    os.remove(file_path)

    # Convert the saved plot to a base64 string to display in HTML
    img_io = BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    img_b64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

    # Return the result page with the prediction and image
    return render_template('result.html', predicted_class=predicted_class, img_b64=img_b64)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
