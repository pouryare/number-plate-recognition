from flask import Flask, render_template, request, jsonify
from tensorflow import keras
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import os
import io
import base64

app = Flask(__name__)

# Define a custom object scope for LeakyReLU
custom_objects = {
    'LeakyReLU': tf.keras.layers.LeakyReLU
}

# Load the model with custom object scope
with tf.keras.utils.custom_object_scope(custom_objects):
    model = keras.models.load_model(os.path.join(os.getcwd(), 'Number-Plate-Recognition.keras'))

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict(image):
    original_size = image.size
    preprocessed_img = preprocess_image(image)
    prediction = model.predict(preprocessed_img)[0]
    
    # Denormalize the predictions
    xmin, xmax, ymin, ymax = prediction
    xmin *= original_size[0]
    xmax *= original_size[0]
    ymin *= original_size[1]
    ymax *= original_size[1]
    
    return {
        'xmin': int(xmin),
        'xmax': int(xmax),
        'ymin': int(ymin),
        'ymax': int(ymax),
        'confidence': float(np.mean(prediction))
    }

def draw_bounding_box(image, bbox):
    draw = ImageDraw.Draw(image)
    draw.rectangle([bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']], outline="red", width=3)
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes))
        result = predict(img)
        
        # Draw bounding box on the image
        img_with_bbox = draw_bounding_box(img, result)
        
        # Convert image to base64 for sending back to client
        buffered = io.BytesIO()
        img_with_bbox.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'prediction': result,
            'image': img_str
        })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)