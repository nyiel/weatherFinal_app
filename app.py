import streamlit as st
from PIL import Image
import torch
import time
from model_utils import load_model, preprocess_image, predict_weather, WEATHER_CLASSES, text_to_speech, get_voice_announcement

# �️ Initialize session state for toggles - MUST BE FIRST
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'voice_enabled' not in st.session_state:
    st.session_state.voice_enabled = False

# �🌐 Page setup
st.set_page_config(
    page_title="Weather Classifier 🌤️",
    page_icon="🌈",
    layout="centered"
)

# 🌍 Language toggle
language = st.selectbox("🌐 Language / اللغة", ["English", "العربية"])

# 🎛️ Modern Settings Section
st.markdown("### ⚙️ Settings")

# Create better spaced columns for modern layout
col_space1, col_dark, col_space2, col_voice, col_space3 = st.columns([0.5, 2, 0.3, 2, 0.5])

# Ensure session state is accessible before using it
dark_mode_state = st.session_state.get('dark_mode', False)
voice_enabled_state = st.session_state.get('voice_enabled', False)

with col_dark:
    # Dark mode toggle with enhanced styling
    dark_icon = "🌙" if not dark_mode_state else "☀️"
    dark_text = "Switch to Dark Mode" if not dark_mode_state else "Switch to Light Mode"
    
    # Create a more prominent button
    if st.button(f"{dark_icon} Mode", key="dark_toggle", help=dark_text, use_container_width=True):
        st.session_state.dark_mode = not dark_mode_state
        st.rerun()
    
    # Modern status indicator
    status_color = "#1976D2" if not dark_mode_state else "#90CAF9"
    status_bg = "#E3F2FD" if not dark_mode_state else "#1E1E1E"
    status_text = "☀️ Light Mode" if not dark_mode_state else "🌙 Dark Mode"
    
    st.markdown(f"""
    <div style="
        background: {status_bg}; 
        color: {status_color}; 
        padding: 8px 16px; 
        border-radius: 20px; 
        text-align: center; 
        font-weight: 600;
        border: 1px solid {status_color}30;
        margin-top: 8px;
    ">
        {status_text}
    </div>
    """, unsafe_allow_html=True)

with col_voice:
    # Voice toggle with enhanced styling
    voice_icon = "🔊" if voice_enabled_state else "🔇"
    voice_text = "Enable Voice Announcements" if not voice_enabled_state else "Disable Voice Announcements"
    
    # Create a more prominent button
    if st.button(f"{voice_icon} Voice", key="voice_toggle", help=voice_text, use_container_width=True):
        st.session_state.voice_enabled = not voice_enabled_state
        st.rerun()
    
    # Modern status indicator
    status_color = "#4CAF50" if voice_enabled_state else "#757575"
    status_bg = "#E8F5E8" if voice_enabled_state else "#F5F5F5"
    status_text = "🔊 Voice ON" if voice_enabled_state else "🔇 Voice OFF"
    
    st.markdown(f"""
    <div style="
        background: {status_bg}; 
        color: {status_color}; 
        padding: 8px 16px; 
        border-radius: 20px; 
        text-align: center; 
        font-weight: 600;
        border: 1px solid {status_color}30;
        margin-top: 8px;
    ">
        {status_text}
    </div>
    """, unsafe_allow_html=True)

# Add a stylish separator
st.markdown("""
<div style="
    height: 2px; 
    background: linear-gradient(90deg, transparent, #667eea, transparent); 
    margin: 20px 0;
    border-radius: 1px;
"></div>
""", unsafe_allow_html=True)

# Set variables based on session state
dark_mode = st.session_state.get('dark_mode', False)
voice_enabled = st.session_state.get('voice_enabled', False)

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

#  Enhanced Colors based on session state
bg_color = "#121212" if dark_mode else "#ffffff"
text_color = "#e0e0e0" if dark_mode else "#000000"
header_color = "#90CAF9" if dark_mode else "#0D47A1"
subheader_color = "#B0BEC5" if dark_mode else "#555"
result_box_color = "#1E1E1E" if dark_mode else "#E3F2FD"
card_bg = "#2D2D2D" if dark_mode else "#F8F9FA"
border_color = "#404040" if dark_mode else "#E0E0E0"

# 💅 Enhanced Custom CSS with Modern Design Standards
st.markdown("""
<style>
/* Root variables for consistent theming */
:root {{
    --primary-color: #667eea;
    --secondary-color: #764ba2;
    --success-color: #4CAF50;
    --warning-color: #FF9800;
    --error-color: #f44336;
    --text-primary: {text};
    --bg-primary: {bg};
    --bg-secondary: {card_bg};
    --border-color: {border};
}}

/* Main app styling with smooth transitions */
.stApp {{
    background: linear-gradient(135deg, {bg} 0%, {card_bg} 100%);
    color: {text};
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}}

/* Modern button styling */
.stButton > button {{
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    color: white !important;
    border: none !important;
    border-radius: 25px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3) !important;
    position: relative !important;
    overflow: hidden !important;
}}

.stButton > button:hover {{
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 30px rgba(102, 126, 234, 0.4) !important;
    background: linear-gradient(135deg, #7c4dff, #536dfe) !important;
}}

.stButton > button:active {{
    transform: translateY(-1px) !important;
    transition: all 0.1s !important;
}}

.stButton > button:focus {{
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.3) !important;
    outline: none !important;
}}

/* Header with gradient text */
.header {{
    font-size: 48px !important;
    font-weight: 800 !important;
    margin-bottom: 20px !important;
    text-align: center !important;
    background: linear-gradient(135deg, {header}, #42A5F5, #7c4dff) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    animation: float 3s ease-in-out infinite !important;
}}

@keyframes float {{
    0%, 100% {{ transform: translateY(0px); }}
    50% {{ transform: translateY(-10px); }}
}}

.subheader {{
    font-size: 20px !important;
    color: {subheader} !important;
    margin-bottom: 35px !important;
    text-align: center !important;
    font-weight: 400 !important;
    opacity: 0.9 !important;
}}

/* Enhanced result box with glassmorphism */
.result-box {{
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 20px !important;
    padding: 30px !important;
    margin-top: 30px !important;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1) !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
}}

.result-box:hover {{
    transform: translateY(-5px) !important;
    box-shadow: 0 30px 80px rgba(0, 0, 0, 0.15) !important;
}}

/* Modern confidence bars with animations */
.confidence-bar {{
    height: 24px !important;
    border-radius: 12px !important;
    margin-bottom: 16px !important;
    text-align: center !important;
    padding: 4px 16px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    color: white !important;
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
    overflow: hidden !important;
}}

.confidence-bar::before {{
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    transition: left 0.5s ease;
}}

.confidence-bar:hover {{
    transform: scale(1.02) !important;
    box-shadow: 0 6px 25px rgba(102, 126, 234, 0.4) !important;
}}

.confidence-bar:hover::before {{
    left: 100%;
}}

/* Enhanced radio buttons */
.stRadio > div {{
    flex-direction: row !important;
    justify-content: center !important;
    gap: 15px !important;
}}

.stRadio > div > label {{
    background: {card_bg} !important;
    border: 2px solid {border} !important;
    border-radius: 15px !important;
    padding: 12px 20px !important;
    margin: 0 !important;
    cursor: pointer !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    font-weight: 500 !important;
}}

.stRadio > div > label:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1) !important;
    border-color: var(--primary-color) !important;
}}

/* Sidebar enhancements */
.css-1d391kg {{
    background: {card_bg} !important;
    border-right: 2px solid {border} !important;
    backdrop-filter: blur(10px) !important;
}}

/* Enhanced status messages */
.stSuccess {{
    background: linear-gradient(135deg, var(--success-color), #45A049) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 500 !important;
    box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3) !important;
}}

.stInfo {{
    background: linear-gradient(135deg, var(--primary-color), #1976D2) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 500 !important;
    box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3) !important;
}}

.stError {{
    background: linear-gradient(135deg, var(--error-color), #d32f2f) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 500 !important;
    box-shadow: 0 4px 15px rgba(244, 67, 54, 0.3) !important;
}}

/* File uploader enhancements */
.stFileUploader {{
    border: 3px dashed {border} !important;
    border-radius: 20px !important;
    background: {card_bg} !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    padding: 20px !important;
}}

.stFileUploader:hover {{
    border-color: var(--primary-color) !important;
    background: {card_bg} !important;
    transform: scale(1.02) !important;
}}

/* Camera input styling */
.stCameraInput {{
    border-radius: 20px !important;
    overflow: hidden !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}}

.stCameraInput:hover {{
    transform: scale(1.02) !important;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15) !important;
}}

/* Selectbox enhancements */
.stSelectbox > div > div {{
    background: {card_bg} !important;
    border: 2px solid {border} !important;
    border-radius: 15px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}}

.stSelectbox > div > div:hover {{
    border-color: var(--primary-color) !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2) !important;
}}

/* Loading spinner enhancement */
.stSpinner {{
    border-top-color: var(--primary-color) !important;
}}

/* Responsive design improvements */
@media (max-width: 768px) {{
    .header {{
        font-size: 36px !important;
    }}
    
    .subheader {{
        font-size: 16px !important;
    }}
    
    .result-box {{
        padding: 20px !important;
        margin-top: 20px !important;
    }}
}}
</style>
""".format(
    bg=bg_color, 
    text=text_color, 
    header=header_color, 
    subheader=subheader_color, 
    box=result_box_color,
    card_bg=card_bg,
    border=border_color
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
