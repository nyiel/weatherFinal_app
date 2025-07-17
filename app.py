import streamlit as st
from PIL import Image
import torch
import time
from model_utils import load_model, preprocess_image, predict_weather, WEATHER_CLASSES, text_to_speech, get_voice_announcement

# 🌐 Page setup
st.set_page_config(
    page_title="Weather Classifier 🌤️",
    page_icon="🌈",
    layout="centered"
)

# 🌍 Language toggle
language = st.selectbox("🌐 Language / اللغة", ["English", "العربية"])

# 🌗 Dark mode and Voice toggles with icon-only interaction
# Initialize session state for toggles
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'voice_enabled' not in st.session_state:
    st.session_state.voice_enabled = False

# Create custom toggle buttons that only respond to icon clicks
col1, col2 = st.columns(2)

with col1:
    # Dark mode toggle
    dark_icon = "🌙" if not st.session_state.dark_mode else "☀️"
    dark_text = "Enable Dark Mode" if not st.session_state.dark_mode else "Enable Light Mode"
    
    if st.button(f"{dark_icon}", key="dark_toggle", help=dark_text):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
        
    # Show dark mode status (message only, no button style)
    if st.session_state.dark_mode:
        st.markdown("🌙 **Dark Mode**")
    else:
        st.markdown("☀️ **Light Mode**")

with col2:
    # Voice toggle  
    voice_icon = "🔊" if st.session_state.voice_enabled else "🔇"
    voice_text = "Voice Enabled" if st.session_state.voice_enabled else "Voice Disabled"
    
    if st.button(f"{voice_icon}", key="voice_toggle", help=voice_text):
        st.session_state.voice_enabled = not st.session_state.voice_enabled
        st.rerun()
        
    # Show voice status (message only, no button style)
    if st.session_state.voice_enabled:
        st.markdown("🔊 **Voice ON**")
    else:
        st.markdown("🔇 **Voice OFF**")

# Set variables based on session state
dark_mode = st.session_state.dark_mode
voice_enabled = st.session_state.voice_enabled

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
        "voice_announcement": "🔊 Voice announcement enabled",
        "voice_playing": "🎵 Playing voice announcement...",
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
        "voice_announcement": "🔊 الإعلان الصوتي مفعل",
        "voice_playing": "🎵 جارٍ تشغيل الإعلان الصوتي...",
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
            max_confidence = probs[pred]

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
            
            # Voice announcement in Arabic
            if voice_enabled:
                # Show voice status
                voice_status = st.empty()
                voice_status.info(f"🔊 {L['voice_announcement']}")
                
                try:
                    # Get voice announcement text in Arabic
                    voice_text = get_voice_announcement(class_name, 'العربية', max_confidence)
                    
                    # Use Arabic language code for TTS
                    lang_code = 'ar'
                    
                    # Update status to show playing
                    voice_status.info(f"🎵 {L['voice_playing']}")
                    
                    # Play voice announcement in Arabic
                    text_to_speech(voice_text, lang_code)
                    
                    # Brief delay to show the playing message
                    time.sleep(1)
                    
                    # Update status to show completion
                    voice_status.success("✅ Voice announcement completed!" if language == "English" else "✅ تم تشغيل الإعلان الصوتي!")
                    
                except Exception as e:
                    voice_status.error(f"❌ Voice error: {str(e)}" if language == "English" else f"❌ خطأ في الصوت: {str(e)}")
                    st.info("💡 Please check your system's text-to-speech settings" if language == "English" else "💡 يرجى التحقق من إعدادات النص إلى كلام")

# 📌 Sidebar
with st.sidebar:
    st.markdown(f"## {L['about_title']}")
    st.markdown(L["about_desc"])
    st.markdown("---")
    st.markdown(f"### {L['details_title']}")
    st.markdown(L["details"])
