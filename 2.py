import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16, InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.resnet50 import decode_predictions as resnet_decode
from tensorflow.keras.applications.vgg16 import decode_predictions as vgg_decode
from tensorflow.keras.applications.inception_v3 import decode_predictions as inception_decode
import numpy as np
from PIL import Image
import time
import zipfile 
import io
import os



# Function to preprocess an image
def preprocess_image(img, model_name):
    # Convert RGBA to RGB
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    elif img.mode == 'L':  # If the image is grayscale, convert to RGB
        img = img.convert('RGB')

    if model_name == 'InceptionV3':
        img = img.resize((299, 299))  # InceptionV3 requires 299x299 images
    else:
        img = img.resize((224, 224))  # Resize to 224x224 for ResNet50 and VGG16

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    if model_name == 'ResNet50':
        img_array = resnet_preprocess(img_array)
    elif model_name == 'VGG16':
        img_array = vgg_preprocess(img_array)
    elif model_name == 'InceptionV3':
        img_array = inception_preprocess(img_array)

    return img_array


# Load the pre-trained models based on user selection
def load_model(model_name):
    if model_name == 'ResNet50':
        return ResNet50(weights='imagenet')
    elif model_name == 'VGG16':
        return VGG16(weights='imagenet')
    elif model_name == 'InceptionV3':
        return InceptionV3(weights='imagenet')

# Function to decode predictions based on the model
def decode_predictions(preds, model_name):
    if model_name == 'ResNet50':
        return resnet_decode(preds, top=3)[0]
    elif model_name == 'VGG16':
        return vgg_decode(preds, top=3)[0]
    elif model_name == 'InceptionV3':
        return inception_decode(preds, top=3)[0]

# App title and subtitle
st.title("Image Classification with ResNet50, VGG16, and InceptionV3")
st.write("Upload an image to classify images with one of the pre-trained models.")

# Model selection
model_name = st.selectbox("Choose a model:", ("ResNet50", "VGG16", "InceptionV3"))

# Model descriptions
model_descriptions = {
    "ResNet50": "ResNet50 is a deep convolutional neural network that is 50 layers deep. It can classify images into 1000 object categories.",
    "VGG16": "VGG16 is a deep convolutional network with 16 layers, known for its simplicity and effectiveness in image classification.",
    "InceptionV3": "InceptionV3 is a convolutional neural network architecture that improves efficiency in image classification using factorized convolutions."
}
st.write(model_descriptions[model_name])

# Load the selected model
model = load_model(model_name)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    img_preprocessed = preprocess_image(img, model_name)
    preds = model.predict(img_preprocessed)
    decoded_preds = decode_predictions(preds, model_name)

    # Display predictions
    st.write("### Predictions:")
    for pred in decoded_preds:
        st.write(f"- **{pred[1].capitalize()}**: {pred[2] * 100:.2f}%")

    # Option to download image and predictions as a zip file
    if st.button("Download Image and Predictions"):
        with io.BytesIO() as buffer:
            with zipfile.ZipFile(buffer, "w") as zf:
                # Save the image in the zip
                img.save("image.png")
                zf.write("image.png")

                # Save predictions as text in the zip
                with open("predictions.txt", "w") as f:
                    for pred in decoded_preds:
                        f.write(f"{pred[1].capitalize()}: {pred[2] * 100:.2f}%\n")
                zf.write("predictions.txt")

            st.download_button(
                label="Download ZIP",
                data=buffer.getvalue(),
                file_name="image_and_predictions.zip",
                mime="application/zip",
            )
            os.remove("image.png")
            os.remove("predictions.txt")


st.markdown("""
    <style>
    /* Background with a classy gradient */
    .main {
        background: linear-gradient(to bottom right, #f4f4f4, #d9d9d9);
        color: #333;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Title styling with a glowing effect and color change */
    .title {
        color: #1f3b4d; /* Deep blue color */
        font-size: 2.8em;
        text-align: center;
        margin-bottom: 20px;
        font-weight: bold;
        border-bottom: 2px solid #ccc;
        padding-bottom: 10px;
        text-shadow: 0 0 10px rgba(31, 59, 77, 0.8); /* Glowing effect */
    }

    /* Subtitle styling */
    .subtitle {
        font-size: 1.4em;
        color: #555;
        text-align: center;
        margin-top: -15px;
        margin-bottom: 30px;
    }

    /* Button styling */
    .stButton>button {
        background-color: #1e90ff; /* Classy blue color */
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 4px;
        font-size: 1.1em;
        transition: background-color 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #4682b4; /* Darker blue on hover */
        box-shadow: 0 0 15px rgba(30, 144, 255, 0.6); /* Glowing effect */
    }

    /* Image caption styling */
    .image-caption {
        text-align: center;
        font-style: italic;
        color: #777;
        margin-top: 10px;
    }

    /* Footer styling */
    .footer {
        position: fixed;
        width: 100%;
        text-align: center;
        color: #aaa;
        font-size: 0.9em;
        text-shadow: 0 0 10px #aaa;
        margin: 20px 0;
        padding: 20px 0;
        position: relative;
        animation: floatHorizontal 3s ease-in-out infinite;
    }

    /* Keyframes for the horizontal floating effect */
    @keyframes floatHorizontal {
        0% { transform: translateX(0px); }
        50% { transform: translateX(10px); }
        100% { transform: translateX(0px); }
    }
    </style>
""", unsafe_allow_html=True)


# Footer HTML with scrolling text
st.markdown("""
    <div class="footer">
        <div class="scrolling-text">Built with Streamlit & TensorFlow</div>
    </div>
""", unsafe_allow_html=True)
