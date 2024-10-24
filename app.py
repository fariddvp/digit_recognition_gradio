import tensorflow as tf
import gradio as gr
from PIL import Image, ImageOps
import numpy as np

# Load the model and print summary for debugging
try:
    model = tf.keras.models.load_model('model.h5')
    print("Model loaded successfully!")
    model.summary()  # This will print the model structure for debugging
except Exception as e:
    print(f"Error loading model: {e}")

def recognize_digit(image):
    if image is not None:
        try:
            # Convert image to PIL if it's in numpy array format
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Convert to grayscale and resize to (28, 28)
            image = image.convert('L').resize((28, 28))
            
            # Invert the image (white background with black digit)
            image = ImageOps.invert(image)
            
            # Convert image to numpy array and normalize it
            image = np.array(image).reshape((1, 28, 28, 1)).astype('float32') / 255

            # Print the processed image shape for debugging
            print(f"Processed image shape: {image.shape}")

            # Make prediction
            prediction = model.predict(image)
            
            # Print the prediction array for debugging
            print(f"Model prediction: {prediction}")

            return {str(i): float(prediction[0][i]) for i in range(10)}
        except Exception as e:
            print(f"Error in processing: {e}")
            return "Error in processing the image."
    else:
        return "No image was provided."

iface = gr.Interface(
    fn=recognize_digit,
    inputs=gr.Image(image_mode='L'),
    outputs=gr.Label(num_top_classes=10),
    live=True
)

iface.launch(share=True)

