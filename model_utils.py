import os
import gdown
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pyttsx3
import threading

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

def text_to_speech(text, language='en'):
    """Convert text to speech with language support."""
    def speak():
        try:
            engine = pyttsx3.init()
            
            # Get available voices
            voices = engine.getProperty('voices')
            
            # Set voice based on language
            if language == 'ar':
                # Try to find Arabic voice
                arabic_voice = None
                for voice in voices:
                    if 'arabic' in voice.name.lower() or 'ar-' in voice.id.lower():
                        arabic_voice = voice.id
                        break
                
                if arabic_voice:
                    engine.setProperty('voice', arabic_voice)
                else:
                    # Fallback to slower speech rate for Arabic text
                    engine.setProperty('rate', 150)
            else:
                # Use English voice (usually default)
                english_voice = None
                for voice in voices:
                    if 'english' in voice.name.lower() or 'en-' in voice.id.lower():
                        english_voice = voice.id
                        break
                
                if english_voice:
                    engine.setProperty('voice', english_voice)
                
                engine.setProperty('rate', 180)
            
            # Set volume
            engine.setProperty('volume', 0.9)
            
            # Speak the text
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            
        except Exception as e:
            print(f"TTS Error: {e}")
    
    # Run TTS in a separate thread to avoid blocking the UI
    thread = threading.Thread(target=speak)
    thread.daemon = True
    thread.start()

def get_voice_announcement(class_name, language='English', confidence=None):
    """Get voice announcement text for the predicted weather."""
    
    # Voice announcements in both languages
    voice_texts = {
        "English": {
            'Cloudy': f"The weather prediction is Cloudy with {confidence:.1f}% confidence. Overcast skies with possible light rain.",
            'Rain': f"The weather prediction is Rain with {confidence:.1f}% confidence. Rain is expected, grab an umbrella!",
            'Shine': f"The weather prediction is Shine with {confidence:.1f}% confidence. Clear skies, great for outdoor activities!",
            'Sunrise': f"The weather prediction is Sunrise with {confidence:.1f}% confidence. Beautiful sunrise or sunset conditions.",
        },
        "العربية": {
            'Cloudy': f"التنبؤ بالطقس هو غائم بثقة {confidence:.1f}%. سماء ملبدة بالغيوم مع احتمال هطول أمطار خفيفة.",
            'Rain': f"التنبؤ بالطقس هو ممطر بثقة {confidence:.1f}%. من المتوقع هطول أمطار، لا تنس المظلة!",
            'Shine': f"التنبؤ بالطقس هو مشمس بثقة {confidence:.1f}%. سماء صافية، طقس مناسب للنشاطات الخارجية!",
            'Sunrise': f"التنبؤ بالطقس هو شروق بثقة {confidence:.1f}%. شروق أو غروب جميل.",
        }
    }
    
    if confidence is None:
        # Simplified version without confidence
        voice_texts["English"][class_name] = voice_texts["English"][class_name].split(" with")[0] + "."
        voice_texts["العربية"][class_name] = voice_texts["العربية"][class_name].split(" بثقة")[0] + "."
    
    return voice_texts.get(language, voice_texts["English"]).get(class_name, "Weather prediction complete.")
