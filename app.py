# -----------------------------
# Libraries
# -----------------------------
import numpy as np
import cv2
import torch
import json
import shap
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import random
import os  # Moved import to top for consistency

# Set seeds for reproducibility
SEED = 4
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Ensure reproducibility for DataLoader (shuffle behavior)
random.seed(SEED)


# -----------------------------
# Source files
# -----------------------------
with open("Output/model_metrics.json", "r") as f:
    metrics_dict = json.load(f)

vgg_model = tf.keras.models.load_model("Output/vgg_model.h5")    # VGG16 model
vgg_accuracy = metrics_dict["vgg_accuracy"]
f1_vgg = metrics_dict["f1_vgg"]


# -----------------------------
# Grad-CAM
# -----------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generate a Grad-CAM heatmap for a given image and model.
    
    Args:
        img_array (numpy.ndarray): Preprocessed image array of shape (1, height, width, channels).
        model (tf.keras.Model): Trained Keras model.
        last_conv_layer_name (str): Name of the last convolutional layer in the model.
        pred_index (int, optional): Index of the class to generate Grad-CAM for. Defaults to None.
    
    Returns:
        numpy.ndarray: Heatmap of shape (height, width).
    """
    # Check if model.output is a list or a single tensor
    if isinstance(model.output, list):
        # If model has multiple outputs, select the first one (modify as needed)
        output = model.output[0]
    else:
        output = model.output

    # Create a model that maps the input image to the activations of the last conv layer and the output
    grad_model = Model(
        inputs=model.inputs, 
        outputs=[model.get_layer(last_conv_layer_name).output, output]
    )
    
    # Compute the gradient of the top predicted class for the input image
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    # Compute gradients with respect to the last conv layer
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Compute the guided gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply each channel in the feature map array by "how important this channel is" with regard to the predicted class
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Apply ReLU to the heatmap to keep only positive activations
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def display_gradcam(img_path, model, last_conv_layer_name):
    """
    Generate and return the Grad-CAM heatmap superimposed on the original image.
    
    Args:
        img_path (str): Path to the image file.
        model (tf.keras.Model): Trained Keras model.
        last_conv_layer_name (str): Name of the last convolutional layer in the model.
    
    Returns:
        numpy.ndarray: Superimposed image with Grad-CAM heatmap.
    """
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Rescale as done in ImageDataGenerator
    
    # Generate heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    
    # Load the original image using OpenCV
    img = cv2.imread(img_path)
    img = cv2.resize(img, (150, 150))
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                              # Convert BGR to RGB for consistency with matplotlib
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))             # Resize the heatmap to match the image size
    heatmap = np.uint8(255 * heatmap)                                       # Rescale heatmap to a range 0-255
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)                  # Apply the heatmap to the image
    superimposed_img = heatmap * 0.4 + img                                  # Superimpose the heatmap on the original image
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')    # Ensure the pixel values are within [0, 255]
    
    # # Optionally, resize the image for better display in Streamlit
    # display_size = (600, 600)  # Adjust the size as needed
    # superimposed_img_resized = cv2.resize(superimposed_img, display_size)
    

    return superimposed_img

# -----------------------------
# LIME Implementation
# -----------------------------
def load_and_preprocess_image(image_path, target_size=(150, 150)):
    """
    Load and preprocess an image for LIME.

    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Desired image size.

    Returns:
        numpy.ndarray: Preprocessed image array.
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_np = np.array(img)
    return img_np

def lime_explanation(image_path, model, target_size=(150, 150), num_samples=1000, top_labels=1, hide_color=0, num_features=10):
    """
    Generate and display LIME explanation for a given image and model side by side with the original image.

    Args:
        image_path (str): Path to the image file.
        model (tf.keras.Model): Trained Keras model.
        target_size (tuple): Desired image size for the model.
        num_samples (int): Number of samples for LIME.
        top_labels (int): Number of top labels to explain.
        hide_color (int): Color to hide the superpixels.
        num_features (int): Number of features (superpixels) to highlight.
    """
    # Load and preprocess the image
    img_np = load_and_preprocess_image(image_path, target_size)
    
    # Define a prediction function for LIME
    def predict_fn(images):
        """
        Prediction function for LIME.

        Args:
            images (list): List of images as numpy arrays.

        Returns:
            numpy.ndarray: Array of probabilities.
        """
        images = np.array(images)
        images = images / 255.0  # Rescale as done in ImageDataGenerator
        predictions = model.predict(images)
        return predictions
    
    # Initialize LIME Image Explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Generate LIME explanation
    explanation = explainer.explain_instance(
        img_np, 
        predict_fn, 
        top_labels=top_labels, 
        hide_color=hide_color, 
        num_samples=num_samples
    )
    
    # Get the top predicted label
    top_pred = explanation.top_labels[0]
    # Assuming binary classification: 0 - Normal, 1 - Pneumonia
    class_mapping = {0: 'Normal', 1: 'Pneumonia'}
    label_name = class_mapping.get(top_pred, f"Class {top_pred}")
    
    # Get image and mask for the top predicted label
    temp, mask = explanation.get_image_and_mask(
        top_pred, 
        positive_only=True, 
        num_features=num_features, 
        hide_rest=False
    )
    
    # Superimpose LIME explanation on the original image
    fig, axes = plt.subplots(figsize=(7, 7))  # Single plot for LIME explanation

    # Display LIME explanation
    axes.imshow(mark_boundaries(temp, mask))
    axes.set_title(f"LIME Explanation")
    axes.axis('off')

    plt.tight_layout()
    st.pyplot(fig)


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.title("Chest X-Ray Pneumonia Classification")

    # Uploading the file
    uploaded_file = st.file_uploader("Upload a Chest X-ray image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        
        temp_image_path = "temp_image.png"
        with open(temp_image_path, "wb") as f:
            f.write(file_bytes)
        
        opencv_image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), 1)
        if opencv_image is None:
            st.error("Error in processing the uploaded image. Please try a different file.")
            return
        
        img_size = (150, 150)
        image_resized = cv2.resize(opencv_image, img_size)
        input_image = image_resized / 255.0

        preds = vgg_model.predict(np.expand_dims(input_image, axis=0))
        prediction_prob = preds[0][0]
        class_label = "Pneumonia" if prediction_prob >= 0.5 else "Normal"

        accuracy_val = 0.859#metrics_dict["vgg_accuracy"]
        f1_val = 0.70#metrics_dict["f1_vgg"]

        # Organizing layout in columns for model info
        st.markdown("<h3 style='text-align: left;'>Model: VGG-16</h3>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<h4 style='text-align: center;'>Accuracy: {accuracy_val*100:.1f}%</h4>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<h4 style='text-align: center;'>F1-Score: {f1_val*100:.1f}%</h4>", unsafe_allow_html=True)

        # Display model prediction and message
        st.subheader("Model Prediction")
        st.markdown(f"**Predicted Class:** {class_label}")
        if class_label == "Normal":
            st.info("No signs of Pneumonia. Consultation with a doctor is likely **not** required. Stay healthy!")
        else:
            st.warning("The model suggests Pneumonia. Please consult with a healthcare professional for further steps.")

        # Display uploaded image
        st.image(opencv_image, caption="Uploaded Chest X-ray")

        # ----------------------------------------------------
        # XAI Methods
        # ----------------------------------------------------
        st.markdown("### Explainability Methods")

        # 1. Grad-CAM
        st.markdown("#### Grad-CAM Explanation")
        st.markdown("""
        **Grad-CAM** highlights the regions that were most influential for the model's decision by producing a heatmap over the image.

        **Red and Yellow**: 
        - These regions indicate the areas where the model has assigned the highest relevance. 
        - They are the regions most responsible for the model's decision and are likely to correspond to abnormalities or features associated with pneumonia.  

        **Blue and Green**: 
        - These areas are less influential in the model's decision-making process. 
        - These regions are not as significant in predicting the class and are less involved in the final prediction.
        """)
        last_conv_layer_name = 'block5_conv3'
        gradcam_result = display_gradcam(temp_image_path, vgg_model, last_conv_layer_name)
        st.image(gradcam_result, caption="Grad-CAM Heatmap", use_container_width=True)

        # 2. LIME
        st.markdown("#### LIME Explanation")
        st.markdown("""
        **LIME** (Local Interpretable Model-agnostic Explanations) explains the prediction by perturbing the input image and observing the changes in predictions.
        
        **Yellow Marking:** 
        - The yellow markings on the image represent the most influential regions for the model's prediction. These areas are the parts of the image that the model considered most important when making its decision. In this case, the highlighted regions are contributing significantly to the model's classification.
        """)
        lime_explanation(temp_image_path, vgg_model, target_size=(150, 150), num_samples=100, top_labels=1, hide_color=0, num_features=10)


if __name__ == "__main__":
    main()
