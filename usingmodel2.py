from keras.models import load_model
from skimage.io import imread
from skimage.transform import resize
import numpy as np

# Load the saved model
loaded_model = load_model('clf-data/keras_model61.h5')

# Function to preprocess a single image for prediction
def preprocess_image(img_path):
    img = imread(img_path)
    
    # Convert to grayscale if the image has multiple channels
    if len(img.shape) == 3:
        img = np.mean(img, axis=-1, keepdims=True).astype(np.uint8)
    
    # Convert to a specific bit depth or data type
    img = img.astype(np.uint8)  # Assuming 8-bit depth
    
    # Resize the image to the same size as used during training
    img = resize(img, (16, 16, 1))  # Adjust the size if you changed target_size during training
    
    return img

# Example usage: Make predictions on a single image
new_image_path = 'clf-data/Amedeo_Modigliani_24.jpg'  # Replace with the path to your new image
preprocessed_image = preprocess_image(new_image_path)

# Reshape the image to match the model's input shape
input_image = np.expand_dims(preprocessed_image, axis=0)

# Make predictions
predictions = loaded_model.predict(input_image)

# Convert predictions to class labels
predicted_class = np.argmax(predictions)

# Get the class labels from the LabelEncoder used during training
class_labels = ['Albrecht_Duhrer', 'Alfred_Sisley', 'Amedeo_Modigliani']

# Get the predicted class label
predicted_label = class_labels[predicted_class]

print(f"The predicted class is: {predicted_label}")
