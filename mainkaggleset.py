import os
from flask import Flask, request, render_template
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('tomato_disease_model (1).h5')

class_indices = {0: "Bacterial_spot",1: "Late_blight",2: "Early_blight",3: "Leaf_Mold",4: "Septoria_leaf_spot",5: "Spider_mites Two-spotted_spider_mite",6: "Target_Spot",7: "Tomato_Yellow_Leaf_Curl_Virus",8: "Tomato_mosaic_virus",9: "healthy",10: "powdery_mildew"}   
class_names = {v: k for k, v in class_indices.items()}  


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(299, 299)) 
    img_array = img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    return img


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

    
    img = load_img(file_path, target_size=(299, 299))  
    img_array = img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = class_indices.get(predicted_class, "Unknown Disease")


   
    plt.imshow(load_img(file_path))  
    plt.axis('off')  
    plt.title(f'Predicted Disease: {predicted_label}')
    plt.savefig('static/result.png')  

   
    os.remove(file_path)

    return render_template('result.html', predicted_label=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)
