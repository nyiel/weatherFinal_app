# Weather Classifier App 🌤️

A deep learning-powered weather classification app with multilingual support and voice announcements.

## Features

### 🧠 AI-Powered Classification
- Uses EfficientNet-B7 model with 97.78% accuracy
- Classifies images into 4 weather types: Cloudy, Rain, Shine, Sunrise
- Real-time confidence scores

### 🌍 Multilingual Support
- English and Arabic UI
- Complete translation for all interface elements
- Language-specific weather tips

### 🔊 Voice Announcements (NEW!)
- Toggle voice announcements on/off
- Speaks predictions in both English and Arabic
- Includes confidence levels in announcements
- Background processing to avoid UI blocking

### 🎨 Modern Interface
- Dark/Light mode toggle
- Responsive design
- File upload or camera capture
- Real-time progress indicators

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

## Voice Feature Usage

1. **Enable Voice**: Toggle the "🔊 Voice Announcements" switch
2. **Select Language**: Choose English or Arabic from the language dropdown
3. **Upload Image**: Use either file upload or camera capture
4. **Get Prediction**: Click "Predict Weather" button
5. **Listen**: The app will announce the result in your selected language

### Voice Announcements Include:
- Weather prediction type
- Confidence percentage
- Weather-specific advice

### Example Announcements:

**English**: *"The weather prediction is Shine with 95.5% confidence. Clear skies, great for outdoor activities!"*

**Arabic**: *"التنبؤ بالطقس هو مشمس بثقة 95.5%. سماء صافية، طقس مناسب للنشاطات الخارجية!"*

## Technical Details

### Voice Technology
- **Library**: pyttsx3 (cross-platform TTS)
- **Languages**: English and Arabic support
- **Processing**: Background threading to prevent UI freezing
- **Voice Selection**: Automatic detection of available system voices

### Model Architecture
- **Base Model**: EfficientNet-B7
- **Input Size**: 224x224 pixels
- **Framework**: PyTorch
- **Classes**: 4 weather categories

## Files Structure

```
weatherFinal_app/
├── app.py              # Main Streamlit application
├── model_utils.py      # Model loading and voice utilities
├── requirements.txt    # Dependencies
├── voice_demo.py      # Voice functionality demo
└── test_voice.py      # Voice testing script
```

## Dependencies

- streamlit
- torch
- torchvision  
- pillow
- gdown
- numpy
- pyttsx3 (for voice functionality)

## Demo Scripts

### Voice Demo
Run the voice demo to test TTS functionality:
```bash
python voice_demo.py
```

### Voice Test
Test voice functionality independently:
```bash
python test_voice.py
```

## Troubleshooting

### Voice Issues
- **No sound**: Check system volume and speakers
- **Wrong language**: Verify system has appropriate TTS voices installed
- **Slow performance**: Voice runs in background to avoid blocking UI

### Common Solutions
- Install additional TTS voices through system settings
- Check firewall settings for audio applications
- Ensure microphone permissions (for camera feature)

## Browser Support

The app works in all modern browsers. For best voice experience:
- Chrome: Full support
- Firefox: Full support  
- Safari: Full support
- Edge: Full support

## Contributing

1. Fork the repository
2. Create feature branch
3. Add voice language support or improve existing features
4. Submit pull request

## License

This project is open source and available under the MIT License.
