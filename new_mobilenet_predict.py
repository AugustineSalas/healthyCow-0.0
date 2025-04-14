import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('best_model.keras')

# Define class labels (for binary classification: 0 = normal, 1 = infected)
class_names = ['normal', 'infected']

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

# Function to predict the class of an image with a confidence score
def predict_image_class(img_path, threshold=0.5):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    # Since our model outputs a single probability value:
    prob = prediction[0][0]
    if prob > threshold:
        label = 'normal'
        confidence = prob * 100
    else:
        label = 'infected'
        confidence = (1 - prob) * 100
    return label, confidence

# Example usage
img_path = 'C:\\Augustine\\Major Project\\Bovine Care\\infected.png'  # Change this to the image you want to test
result, confidence = predict_image_class(img_path)
print(f'The cow is: {result} with confidence: {confidence:.2f}%')
