# Import necessary libraries
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from lime import lime_image
from skimage.segmentation import mark_boundaries, slic
from PIL import Image
import io
import base64

# Initialize the FastAPI application
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the fine-tuned model
model = tf.keras.models.load_model(r"..\model\MP_VGG16-FineTunning-Modified-V3.h5")

# Define a function to preprocess images
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize to expected size
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)  # Normalize the image

# Define the prediction function for LIME
def predict_fn(images):
    return model.predict(images)

# Function to visualize images with superpixels
def visualize_superpixels(img_data, n_segments=50, compactness=10):
    segments = slic(img_data, n_segments=n_segments, compactness=compactness, sigma=1)
    return segments

# Function to convert plots to bytes and then to base64
def plot_to_base64(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')  # Convert to base64

# API endpoint for image upload and prediction
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Load and preprocess the image
    original_image = Image.open(file.file)
    preprocessed_image = preprocess_image(original_image)
    img_data = np.array(original_image)

    # Explain the prediction
    explainer = lime_image.LimeImageExplainer()
    predictions = predict_fn(preprocessed_image)

    # Get explanation
    explanation = explainer.explain_instance(
        image=img_data,
        classifier_fn=predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=250
    )

    label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        label=label,
        positive_only=False,
        hide_rest=False,
        num_features=10
    )

    # Visualizations
    # Original Image
    plt.figure(figsize=(5, 5))
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis('off')
    original_image_base64 = plot_to_base64(plt.gcf())
    plt.close()


    # Image with Superpixels
    plt.figure(figsize=(5, 5))
    segments = visualize_superpixels(img_data)
    plt.imshow(mark_boundaries(img_data / 255.0, segments))
    plt.title("Image with Superpixels")
    plt.axis('off')
    superpixels_image_base64 = plot_to_base64(plt.gcf())
    plt.close()

    # LIME Explanation
    plt.figure(figsize=(5, 5))
    plt.imshow(mark_boundaries(temp / 255.0, mask))
    plt.title("LIME Positive Contributions and Negative")
    plt.axis('off')
    lime_explanation_base64 = plot_to_base64(plt.gcf())
    plt.close()

    # Visualize positive contributions only
    temp_pos, mask_pos = explanation.get_image_and_mask(label=label, positive_only=True, hide_rest=False, num_features=5)
    plt.figure(figsize=(5, 5))
    plt.imshow(mark_boundaries(temp_pos / 255.0, mask_pos))
    plt.title("LIME Positive Contributions")
    plt.axis('off')
    lime_positive_base64 = plot_to_base64(plt.gcf())
    plt.close()

    # Distribution of Feature Importance
    feature_importance = explanation.local_exp[label]
    features, weights = zip(*feature_importance)
    sorted_indices = np.argsort(weights)[::-1]
    plt.figure(figsize=(15, 5))
    plt.bar(range(len(weights)), np.array(weights)[sorted_indices], color='blue')
    plt.xticks(range(len(weights)), np.array(features)[sorted_indices], rotation=90)
    plt.title("Feature Importance of Superpixels")
    plt.xlabel("Superpixels")
    plt.ylabel("Weight")
    plt.grid(axis='y')
    feature_importance_base64 = plot_to_base64(plt.gcf())
    plt.close()

    # Highlighting Top Contributing Superpixels
    top_n = 5  # Number of top superpixels to highlight
    top_features = sorted_indices[:top_n]
    highlighted_mask = np.zeros_like(mask)

    for i in top_features:
        highlighted_mask += (mask == i).astype(int)

    plt.figure(figsize=(5, 5))
    plt.imshow(original_image)
    plt.imshow(highlighted_mask, alpha=0.5, cmap='jet')
    plt.title(f"Top {top_n} Contributing Superpixels Highlighted")
    plt.axis('off')
    top_contributing_base64 = plot_to_base64(plt.gcf())
    plt.close()

    # Perturbed Images
    def perturb_image(image, segments, num_superpixels):
        perturbed_images = []
        for i in range(num_superpixels):
            temp_image = image.copy()
            temp_image[segments == i] = 0  # Zero out current superpixel
            perturbed_images.append(temp_image)
        return perturbed_images

    num_superpixels_to_display = 4
    perturbed_images = perturb_image(preprocessed_image[0], segments, num_superpixels_to_display)

    perturbed_images_base64 = []
    for i, perturbed_image in enumerate(perturbed_images):
        plt.figure(figsize=(5, 5))
        plt.imshow(perturbed_image / 255.0)
        plt.title(f"Perturbed Image {i + 1}")
        plt.axis('off')
        perturbed_images_base64.append(plot_to_base64(plt.gcf()))
        plt.close()



    # Overlaying LIME Mask on Original Image
    plt.figure(figsize=(5, 5))
    plt.imshow(original_image)
    plt.imshow(mark_boundaries(np.zeros_like(original_image), mask), alpha=0.5)
    plt.title("LIME Mask Overlay on Original Image")
    plt.axis('off')
    mask_overlay_base64 = plot_to_base64(plt.gcf())
    plt.close()

    print(predictions)

    return {
        "predictions": predictions.tolist(),
        "images": {
            "original_image": f"data:image/png;base64,{original_image_base64}",
            "superpixels_image": f"data:image/png;base64,{superpixels_image_base64}",
            "lime_explanation": f"data:image/png;base64,{lime_explanation_base64}",
            "lime_positive": f"data:image/png;base64,{lime_positive_base64}",
            "feature_importance": f"data:image/png;base64,{feature_importance_base64}",
            "top_contributing": f"data:image/png;base64,{top_contributing_base64}",
            "perturbed_images": [f"data:image/png;base64,{img_base64}" for img_base64 in perturbed_images_base64],
            "mask_overlay": f"data:image/png;base64,{mask_overlay_base64}",
        }
    }

# To run the FastAPI application, use the command:
# uvicorn main:app --reload
