import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Define your weather classes
WEATHER_CLASSES = ['Cloudy', 'Rain', 'Shine', 'Sunrise']

def load_model(model_path):
    """Load the trained PyTorch model"""
    model = models.efficientnet_b7(pretrained=False)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.features[-2:].parameters():
        param.requires_grad = True
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, len(WEATHER_CLASSES))
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image):
    """Preprocess the image for model prediction"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)
    return image

def predict_weather(model, image):
    """Predict weather from image using the model"""
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
    return predicted.item(), probabilities.numpy()
