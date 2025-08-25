import os
from flask import Flask, request, render_template
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'  # Folder to save uploaded images
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = load_model('tomato_disease_model.h5')

# Map class indices to class names
class_indices = {0: "healthy", 1: "yellow_leaf_curl", 2: "early_blight", 3: "septoria_leaf_mold", 4: "mosaic", 5: "2_spotted_spider_mite", 6: "leaf_miner"}  # Update with actual classes
class_names = {v: k for k, v in class_indices.items()}  # Reverse mapping

# Preprocess image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(150, 150))  # Match your model's input size
    img_array = img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img, img_array

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')  # Render the upload page

@app.route('/predict', methods=['POST'])
def predict():
    # Handling uploaded file
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Saving the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Preprocessing and predicting
    img, img_array = preprocess_image(file_path)
    predictions = model.predict(img_array) #COMPARING THE MODEL WITH IMAGE

    # Get the predicted class
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = class_indices.get(predicted_class, "Unknown Disease")

    # Display the result
    plt.imshow(img)  # Display the uploaded image
    plt.axis('off')  # Hide the axis
    plt.title(f'Predicted Disease: {predicted_label}')
    result_image_path = os.path.join('static', 'result.png')
    plt.savefig(result_image_path)  # Save the result as an image for display

    # Cleanup uploaded image
    os.remove(file_path)

    return render_template('result.html', predicted_label=predicted_label, result_image_path=result_image_path)

if __name__ == '__main__':
    app.run(debug=True)
