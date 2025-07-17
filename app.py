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

# 🌍 Language toggle
language = st.selectbox("🌐 Language / اللغة", ["English", "العربية"])

# 🗣️ Translation dictionary
T = {
    "English": {
        "title": "⛅ Weather Classifier",
        "subtitle": "Upload or capture a sky image to predict the weather condition.",
        "method_label": "Input method:",
        "upload": "📁 Upload",
        "camera": "📷 Camera",
        "upload_prompt": "Upload your sky image:",
        "camera_prompt": "Take a picture:",
        "predict_button": "🔮 Predict Weather",
        "analyzing": "Analyzing the sky...",
        "prediction": "🌤️ Prediction",
        "confidence": "Confidence Levels:",
        "tips": {
            'Cloudy': "☁️ Overcast skies. Possible light rain.",
            'Rain': "🌧️ Rain expected. Grab an umbrella!",
            'Shine': "☀️ Clear skies. Great for outdoor activities!",
            'Sunrise': "🌅 Beautiful sunrise or sunset conditions.",
        },
        "about_title": "🛠️ About This App",
        "about_desc": """
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
""",
        "details_title": "📊 Model Details",
        "details": """
- Input size: 224x224  
- Framework: PyTorch  
- Fine-tuned on weather dataset  
"""
    },
    "العربية": {
        "title": "⛅ مصنف الطقس",
        "subtitle": "قم بتحميل أو التقاط صورة للسماء للتنبؤ بحالة الطقس.",
        "method_label": "طريقة الإدخال:",
        "upload": "📁 تحميل",
        "camera": "📷 كاميرا",
        "upload_prompt": "قم بتحميل صورة السماء:",
        "camera_prompt": "التقط صورة:",
        "predict_button": "🔮 تنبؤ بالطقس",
        "analyzing": "جارٍ تحليل السماء...",
        "prediction": "🌤️ التنبؤ",
        "confidence": "مستويات الثقة:",
        "tips": {
            'Cloudy': "☁️ سماء ملبدة بالغيوم. احتمال هطول أمطار خفيفة.",
            'Rain': "🌧️ من المتوقع هطول أمطار. لا تنس المظلة!",
            'Shine': "☀️ سماء صافية. طقس مناسب للنشاطات الخارجية!",
            'Sunrise': "🌅 شروق أو غروب جميل.",
        },
        "about_title": "🛠️ حول هذا التطبيق",
        "about_desc": """
يستخدم هذا التطبيق نموذج تعلم عميق لتصنيف صور السماء إلى 4 أنواع من الطقس:

- ☁️ غائم  
- 🌧️ ممطر  
- ☀️ مشمس  
- 🌅 شروق / غروب  

**طريقة الاستخدام:**
1. قم بتحميل أو التقاط صورة للسماء  
2. اضغط على **تنبؤ بالطقس**  
3. عرض النتائج مع مستويات الثقة

*النموذج: EfficientNet-B7 (دقة 97.78%)*
""",
        "details_title": "📊 تفاصيل النموذج",
        "details": """
- حجم الإدخال: 224x224  
- الإطار: PyTorch  
- مدرب على مجموعة بيانات الطقس  
"""
    }
}

# 🌗 Dark mode toggle
dark_mode = st.toggle("🌙 Dark Mode", value=False)

# 🎨 Colors
bg_color = "#121212" if dark_mode else "#ffffff"
text_color = "#e0e0e0" if dark_mode else "#000000"
header_color = "#90CAF9" if dark_mode else "#0D47A1"
subheader_color = "#B0BEC5" if dark_mode else "#555"
result_box_color = "#1E1E1E" if dark_mode else "#E3F2FD"

# 💅 Custom CSS
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
    bg=bg_color, text=text_color, header=header_color, subheader=subheader_color, box=result_box_color
), unsafe_allow_html=True)

# 🧠 Load model
@st.cache_resource
def load_cached_model():
    return load_model('best_model.pth')

model = load_cached_model()

# 🌐 Localized labels
L = T[language]

# ⛅ Title
st.markdown(f'<div class="header">{L["title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="subheader">{L["subtitle"]}</div>', unsafe_allow_html=True)

# 📤 Input method
col1, col2 = st.columns(2)
with col1:
    method = st.radio(L["method_label"], (L["upload"], L["camera"]), horizontal=True)

image = None
if method == L["upload"]:
    file = st.file_uploader(L["upload_prompt"], type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if file:
        image = Image.open(file)
else:
    cam = st.camera_input(L["camera_prompt"], label_visibility="collapsed")
    if cam:
        image = Image.open(cam)

# 🖼️ Predict
if image is not None:
    st.image(image, caption="📷", use_container_width=True)

    if st.button(L["predict_button"], use_container_width=True):
        with st.spinner(L["analyzing"]):
            time.sleep(1)
            img_tensor = preprocess_image(image)
            pred, probs = predict_weather(model, img_tensor)
            class_name = WEATHER_CLASSES[pred]

            placeholder = st.empty()
            with placeholder.container():
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f"## {L['prediction']}: **{class_name}**")
                st.markdown(f"#### {L['confidence']}")

                for i, prob in enumerate(probs):
                    st.markdown(f"**{WEATHER_CLASSES[i]}**")
                    st.markdown(
                        f'<div class="confidence-bar" style="width: {prob}%">{prob:.1f}%</div>',
                        unsafe_allow_html=True
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            st.success(L["tips"][class_name])

# 📌 Sidebar
with st.sidebar:
    st.markdown(f"## {L['about_title']}")
    st.markdown(L["about_desc"])
    st.markdown("---")
    st.markdown(f"### {L['details_title']}")
    st.markdown(L["details"])
