# Import necessary libraries
import os
from dotenv import load_dotenv
# Load environment variables from the .env file
load_dotenv()
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
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from groq import Groq

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

def extract_lime_explanation(explanation):
    # Get the top labels and superpixel weights
    top_labels = explanation.top_labels
    lime_data = []

    for label in top_labels:
        # Get the image and mask for the current label
        temp, mask = explanation.get_image_and_mask(label=label, positive_only=False, hide_rest=False, num_features=10)

        # Extract the superpixels and their importance scores
        weights = explanation.local_exp[label]  # list of (superpixel_id, weight) tuples

        # Store superpixels with their importance scores
        superpixels = [{'superpixel_id': sp_id, 'importance_score': weight} for sp_id, weight in weights]
        lime_data.append({'label': label, 'superpixels': superpixels})

    return lime_data


def explain_with_LLM(lime_explanation_base64,prediction, superpixels):

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    # LLaMA 3.2 90b Vision Model Prompt: LIME explanation and image input
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "This fundus image is accompanied by a LIME-generated explanation to assist in diagnosing glaucoma. "
                            f"The model predicted a probability of {prediction}, and the LIME explanation highlights key regions in the image. "
                            "\n\nThreshold for interpreting the model's prediction:"
                            "\n- If the probability is **less than 0.5**, the diagnosis is normal (healthy eyes)."
                            "\n- If the probability is **greater than or equal to 0.5**, the diagnosis suggests glaucoma."
                            "\n\n### LIME Explanation Breakdown:"
                            "\n- **POSITIVE Contribution (Green regions)**: Areas that contribute to an **increased probability of glaucoma**."
                            "\n- **NEGATIVE Contribution (Red regions)**: Areas that contribute to a **decreased probability of glaucoma**."
                            "\n\n### Task Instructions (Respond in a clear and consistent format):"
                            "\n1. **Interpret the model’s prediction**: Based on the provided probability, determine if the diagnosis is normal or glaucoma."
                            "\n2. **Identify Key Regions**: Explain which areas of the image are highlighted by LIME (e.g., optic disc, retinal nerve fiber thinning)."
                            "\n3. **Significance of LIME’s Contributions**: Discuss the importance of the green (positive) and red (negative) superpixels in relation to glaucoma features."
                            "\n4. **Conclude the Explanation**: Summarize the model’s prediction and the relevance of the LIME explanation to clinical decision-making."
                            "\n\nPlease provide a detailed, consistent, and structured response following these points."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{lime_explanation_base64}"},
                    },
                ],
            }
        ],
        model="llama-3.2-90b-vision-preview",
    )

    vision_model_response =chat_completion.choices[0].message.content

    llm = ChatGroq(
    temperature=0,
    groq_api_key=os.environ.get("GROQ_API_KEY"),
    model_name="llama-3.1-70b-versatile"
    )

    # LLaMA 3.1 70B Explanation Prompt: Summarizing the vision model's output
    explanation_prompt = """
    You are provided with a LIME explanation of an image classification model, which predicts glaucoma. The explanation includes superpixels indicating areas of the image that have a positive or negative influence on the prediction.
    ### Key Details from the LIME Explanation:
    {vision_model_response}

    ### Your task is to:

    1. {superpixels},Summarize the superpixels that had the highest importance in the prediction, both positive and negative regions.
    2. Identify which regions of the eye (e.g., optic disc, retinal nerve fiber layer) were influenced by these superpixels and how they correlate with known features of glaucoma.
    3. Explain the clinical relevance of the regions with positive and negative importance scores in the context of glaucoma diagnosis.
    4. Conclude the LIME explanation and the model’s prediction and its potential impact on the diagnosis.
    5. Explain all in detail.
    ### Response Format:
    Please write a clear, concise explanation. Use complete sentences, avoid technical jargon, and ensure that the response is easy to understand by medical professionals.
    Only return the valid JSON.
    ### VALID JSON (NO PREAMBLE):
    """

    jsonformat = PromptTemplate.from_template(explanation_prompt)
    groq_explanation_chain = jsonformat | llm

    # Invoke the chain
    llm_explanation = groq_explanation_chain.invoke(input={'superpixels':superpixels,'vision_model_response':vision_model_response})
    return llm_explanation

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

    '''GENERATE EXPLANATION THROUGH LIME (Local Interpretable Model-Agnostic Explanations)'''

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

    '''GENERATE EXPLANATION THROUGH LLM (LLaMA 3.1 70B)'''

    # Extract LIME explanation details
    lime_explanation_data = extract_lime_explanation(explanation)

    # Generate explanation using ChatGroq (LLaMA 70B)
    groq_explanation = explain_with_LLM(
        lime_explanation_base64,
        predictions[0][0],
        superpixels=lime_explanation_data[0]['superpixels']
    )

    # Parse explanation from JSON
    json_parser = JsonOutputParser()
    groq_explanation_parsed = json_parser.parse(groq_explanation.content)

    # Visualize the image with annotated superpixel importance
    segments = visualize_superpixels(img_data)
    fig, ax = plt.subplots(figsize=(10, 10))
    image_with_boundaries = mark_boundaries(img_data / 255.0, segments, color=(255, 255, 0))
    ax.imshow(image_with_boundaries)

    # Annotate superpixels with importance scores
    sorted_superpixels = sorted(
        [(sp['superpixel_id'], sp['importance_score']) for sp in lime_explanation_data[0]['superpixels']],
        key=lambda x: abs(x[1]), reverse=True
    )
    for label, importance_score in sorted_superpixels:
        coords = np.column_stack(np.where(segments == label))
        center = coords.mean(axis=0).astype(int)
        if 0 <= center[0] < img_data.shape[0] and 0 <= center[1] < img_data.shape[1]:
            color = (0, 1, 0) if importance_score > 0 else (1, 0, 0)
            ax.text(center[1], center[0], f'{label}: {importance_score:.2f}',
                    color='black', fontsize=12, ha='center', va='center', backgroundcolor=color)

    ax.axis('off')
    plt.title("Superpixels with Importance Scores Highlighted")
    superpixel_importance_base64 = plot_to_base64(fig)
    plt.close()

    return {
        "predictions": predictions.tolist(),
        "groq_explanation": groq_explanation_parsed,
        "images": {
            "original_image": f"data:image/png;base64,{original_image_base64}",
            "superpixels_image": f"data:image/png;base64,{superpixels_image_base64}",
            "lime_explanation": f"data:image/png;base64,{lime_explanation_base64}",
            "lime_positive": f"data:image/png;base64,{lime_positive_base64}",
            "top_contributing": f"data:image/png;base64,{top_contributing_base64}",
            "perturbed_images": [f"data:image/png;base64,{img_base64}" for img_base64 in perturbed_images_base64],
            "mask_overlay": f"data:image/png;base64,{mask_overlay_base64}",
            "superpixel_importance": f"data:image/png;base64,{superpixel_importance_base64}"
        }
    }

# To run the FastAPI application, use the command:
# uvicorn main:app --reload
