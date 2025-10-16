import os
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from transformers import CLIPProcessor, CLIPModel

os.environ["CUDA_VISIBLE_DEVICES"]= '0'
# Load CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


#####################################################################

n_clusters = 7
root_directory = "/path/to/input_directory"
output_directory ="/path/to/output_directory"
#####################################################################

# Function to load and preprocess images
def load_images_and_labels(root_directory, target_size=(224, 224)):
    images = []
    labels = []
    for class_label in os.listdir(root_directory):
        class_dir = os.path.join(root_directory, class_label)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                if file_name.lower().endswith(('png', 'jpg', 'jpeg')):
                    try:
                        img = Image.open(file_path).convert("RGB")
                        images.append(img)
                        labels.append(class_label)
                    except Exception as e:
                        print(f"Could not process {file_name}: {e}")
    return images, labels

# Function to extract CLIP features
def extract_features(images):
    preprocessed_images = clip_processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = clip_model.get_image_features(**preprocessed_images)
    return features.cpu().numpy()

import os
import shutil
from pathlib import Path

# Function to save images into subdirectories based on predicted labels
def save_images_by_predicted_label(images, predicted_labels, output_directory):
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    for img, predicted_label in zip(images, predicted_labels):
        # Create subdirectory for the predicted label
        label_dir = os.path.join(output_directory, str(predicted_label))
        Path(label_dir).mkdir(parents=True, exist_ok=True)

        # Save the image
        img_index = len(os.listdir(label_dir))  # Avoid overwriting existing images
        img.save(os.path.join(label_dir, f"image_{img_index + 1}.png"))
    print(f"Images saved in {output_directory}.")



# Load images and labels
print("Loading images and labels...")
images, labels = load_images_and_labels(root_directory)
if len(images) == 0:
    raise ValueError("No images found in the specified directory.")

# Encode labels as integers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Extract features
print("Extracting features...")
features = extract_features(images)


# Apply k-means clustering to find pseudo-labels
print("Applying k-means clustering...")

print('Total images : ', len(images))
print('Total labels : ', n_clusters)


kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(features)


save_images_by_predicted_label(images, kmeans_labels, output_directory)
