from flask import Flask, request, jsonify
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the pickled model
model_filename = 'fruit_detection_model.pkl'
with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)

# Define the class labels
# Assuming train_generator is defined elsewhere in the code
class_labels = list(train_generator.class_indices.keys())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the POST request
        image_file = request.files['image']
        
        # Load the image and preprocess it for the model
        img_width, img_height = 224, 224  # Replace with your desired image dimensions
        img = image.load_img(image_file, target_size=(img_width, img_height))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image
        
        # Make predictions
        predictions = model.predict(img_array)
        
        # Get the predicted class label
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_labels[predicted_class_index]
        
        # Prepare the response
        response = {
            'predicted_class': predicted_class,
            'confidence': float(predictions[0][predicted_class_index])
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
