import os
import gdown
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Define weather class labels
WEATHER_CLASSES = ['Cloudy', 'Rain', 'Shine', 'Sunrise']

def load_model(model_path='best_model.pth'):
    """Load the trained PyTorch model; download from Google Drive if needed."""

    # Original link: https://drive.google.com/file/d/1hZCVZw1vJXUYODVLB-Ko76tLxDPe_4n8/view?usp=sharing
    # Extracted file ID:
    file_id = '1hZCVZw1vJXUYODVLB-Ko76tLxDPe_4n8'
    gdrive_url = f'https://drive.google.com/uc?id={file_id}'

    if not os.path.exists(model_path):
        try:
            print("📥 Downloading model from Google Drive...")
            gdown.download(gdrive_url, model_path, quiet=False)
        except Exception as e:
            raise RuntimeError(
                f"❌ Failed to download model: {e}\n"
                f"💡 Make sure the file is shared as 'Anyone with the link can view'.\n"
                f"🔗 Your shared link: https://drive.google.com/file/d/{file_id}/view?usp=sharing"
            )

    # Initialize EfficientNet-B7 model
    model = models.efficientnet_b7(pretrained=False)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.features[-2:].parameters():
        param.requires_grad = True

    # Modify classifier for your 4 weather classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, len(WEATHER_CLASSES))
    )

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image):
    """Apply transformations to an image before prediction."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

def predict_weather(model, image):
    """Predict weather category from image."""
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
    return predicted.item(), probabilities.numpy()
