import streamlit as st
from PIL import Image
import numpy as np
import io
import torch
from model_utils import load_model, preprocess_image, predict_weather, WEATHER_CLASSES

# ✅ This must be the FIRST Streamlit command
st.set_page_config(
    page_title="Weather Prediction App",
    page_icon="⛅",
    layout="centered",
    initial_sidebar_state="auto"
)

# Custom CSS for styling
st.markdown("""
<style>
    .header {
        font-size: 36px !important;
        font-weight: bold !important;
        color: #1E88E5 !important;
        margin-bottom: 20px !important;
    }
    .subheader {
        font-size: 20px !important;
        color: #424242 !important;
        margin-bottom: 30px !important;
    }
    .result-box {
        border-radius: 10px;
        padding: 20px;
        background-color: #E3F2FD;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .confidence-bar {
        height: 20px;
        border-radius: 5px;
        margin-bottom: 10px;
        background: linear-gradient(90deg, #64B5F6 0%, #1E88E5 100%);
        color: white;
        padding: 5px;
        text-align: right;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stRadio > div {
        flex-direction: row !important;
    }
    .stRadio > div > label {
        margin-right: 20px !important;
    }
</style>
""", unsafe_allow_html=True)

# Load the model (cached to avoid reloading every time)
@st.cache_resource
def load_cached_model():
    return load_model('best_model.pth')

model = load_cached_model()

# App header
st.markdown('<div class="header">⛅ Weather Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Upload or capture an image of the sky to predict weather conditions</div>', unsafe_allow_html=True)

# Image input options
col1, col2 = st.columns(2)
with col1:
    option = st.radio(
        "Select input method:",
        ("📁 Upload image", "📷 Use camera"),
        horizontal=True
    )

image = None

if option == "📁 Upload image":
    uploaded_file = st.file_uploader(
        "Choose a sky image...", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
else:
    st.write("")
    camera_image = st.camera_input("Take a picture of the sky", label_visibility="collapsed")
    if camera_image is not None:
        image = Image.open(camera_image)

# Display and process the image
if image is not None:
    st.image(
        image, 
        caption="Your sky image",
        use_container_width=True,
        width=300
    )
    
    if st.button('🔮 Predict Weather', use_container_width=True):
        with st.spinner('Analyzing sky conditions...'):
            processed_image = preprocess_image(image)
            prediction, probabilities = predict_weather(model, processed_image)

            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.markdown(f"### 🌤️ Predicted Weather: **{WEATHER_CLASSES[prediction]}**")

            st.markdown("#### Confidence Levels:")
            for i, prob in enumerate(probabilities):
                st.markdown(f"**{WEATHER_CLASSES[i]}**")
                st.markdown(
                    f'<div class="confidence-bar" style="width: {prob}%">{prob:.1f}%</div>',
                    unsafe_allow_html=True
                )

            st.markdown('</div>', unsafe_allow_html=True)

            # Extra weather tips
            weather_info = {
                'Cloudy': "☁️ Overcast skies with possible light precipitation.",
                'Rain': "🌧️ Expect rainfall — don't forget your umbrella!",
                'Shine': "☀️ Clear skies with sunshine — perfect outdoor weather!",
                'Sunrise': "🌅 Beautiful sunrise or sunset conditions."
            }
            st.info(weather_info[WEATHER_CLASSES[prediction]])

# Sidebar content
with st.sidebar:
    st.markdown("## About This App")
    st.markdown("""
This app uses a deep learning model to predict weather conditions from sky images.

**How to use:**
1. Upload or capture a sky image.
2. Click the **Predict Weather** button.
3. View the predicted condition and confidence levels.

**Tips:**
- Use clear sky photos
- Avoid obstructions (trees, buildings)
- Make sure sky is centered

*Model: EfficientNet-B7 — 97.78% validation accuracy*
""")
    st.markdown("---")
    st.markdown("**Model Details**")
    st.markdown("""
- Classes: Cloudy, Rain, Shine, Sunrise  
- Input size: 224x224 pixels  
- Pre-trained on ImageNet  
""")
