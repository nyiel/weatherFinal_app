import streamlit as st
from PIL import Image
import torch
import time
from model_utils import load_model, preprocess_image, predict_weather, WEATHER_CLASSES

# 🌐 Page setup
st.set_page_config(
    page_title="Weather Classifier 🌤️",
    page_icon="🌈",
    layout="centered"
)

# 🌗 Dark mode toggle
dark_mode = st.toggle("🌙 Dark Mode", value=False)

# 🎨 Theme-based colors
bg_color = "#121212" if dark_mode else "#ffffff"
text_color = "#e0e0e0" if dark_mode else "#000000"
header_color = "#90CAF9" if dark_mode else "#0D47A1"
subheader_color = "#B0BEC5" if dark_mode else "#555"
result_box_color = "#1E1E1E" if dark_mode else "#E3F2FD"

# 💅 Custom CSS with .format() to avoid tokenize error
st.markdown("""
<style>
body {{
    background-color: {bg};
    color: {text};
}}

.header {{
    font-size: 40px !important;
    font-weight: 700 !important;
    color: {header} !important;
    margin-bottom: 10px !important;
}}

.subheader {{
    font-size: 18px !important;
    color: {subheader} !important;
    margin-bottom: 30px !important;
}}

.result-box {{
    background-color: {box};
    border-radius: 12px;
    padding: 20px;
    margin-top: 25px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}}

.confidence-bar {{
    height: 18px;
    border-radius: 8px;
    margin-bottom: 8px;
    text-align: right;
    padding: 2px 8px;
    font-weight: bold;
    color: white;
    background: linear-gradient(90deg, #2196F3 0%, #1976D2 100%);
}}

.stRadio > div {{
    flex-direction: row !important;
}}

.stRadio > div > label {{
    margin-right: 20px !important;
}}
</style>
""".format(
    bg=bg_color,
    text=text_color,
    header=header_color,
    subheader=subheader_color,
    box=result_box_color
), unsafe_allow_html=True)

# 🧠 Load model
@st.cache_resource
def load_cached_model():
    return load_model('best_model.pth')

model = load_cached_model()

# ⛅ App title
st.markdown('<div class="header">⛅ Weather Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Upload or capture a sky image to predict the weather condition.</div>', unsafe_allow_html=True)

# 📤 Input method
col1, col2 = st.columns(2)
with col1:
    method = st.radio("Input method:", ("📁 Upload", "📷 Camera"), horizontal=True)

image = None
if method == "📁 Upload":
    file = st.file_uploader("Upload your sky image:", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if file:
        image = Image.open(file)
else:
    cam = st.camera_input("Take a picture:", label_visibility="collapsed")
    if cam:
        image = Image.open(cam)

# 🖼️ Show image + Prediction
if image is not None:
    st.image(image, caption="📷 Input Image", use_container_width=True)

    if st.button("🔮 Predict Weather", use_container_width=True):
        with st.spinner("Analyzing the sky..."):
            time.sleep(1)  # Simulate animation delay
            img_tensor = preprocess_image(image)
            pred, probs = predict_weather(model, img_tensor)
            class_name = WEATHER_CLASSES[pred]

            placeholder = st.empty()
            with placeholder.container():
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f"## 🌤️ Prediction: **{class_name}**")
                st.markdown("#### Confidence Levels:")

                for i, prob in enumerate(probs):
                    st.markdown(f"**{WEATHER_CLASSES[i]}**")
                    st.markdown(
                        f'<div class="confidence-bar" style="width: {prob}%">{prob:.1f}%</div>',
                        unsafe_allow_html=True
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            # 💡 Tips
            tips = {
                'Cloudy': "☁️ Overcast skies. Possible light rain.",
                'Rain': "🌧️ Rain expected. Grab an umbrella!",
                'Shine': "☀️ Clear skies. Great for outdoor activities!",
                'Sunrise': "🌅 Beautiful sunrise or sunset conditions.",
            }
            st.success(tips[class_name])

# 📌 Sidebar content
with st.sidebar:
    st.markdown("## 🛠️ About This App")
    st.markdown("""
This app uses a deep learning model to classify sky images into 4 weather types:

- ☁️ Cloudy  
- 🌧️ Rain  
- ☀️ Shine  
- 🌅 Sunrise  

**How to use:**
1. Upload or take a photo of the sky  
2. Click **Predict Weather**  
3. View results with confidence levels

*Model: EfficientNet-B7 (97.78% accuracy)*
""")
    st.markdown("---")
    st.markdown("### 📊 Model Details")
    st.markdown("""
- Input size: 224x224  
- Framework: PyTorch  
- Fine-tuned on weather dataset  
""")