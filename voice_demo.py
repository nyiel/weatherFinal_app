#!/usr/bin/env python3
"""Demo script for voice functionality without Streamlit dependencies"""

import pyttsx3
import threading
import time

def demo_voice_functionality():
    """Demonstrate the voice functionality with sample weather predictions."""
    
    print("🔊 Weather Classifier Voice Demo")
    print("=" * 40)
    
    def speak_text(text, language_code='en'):
        """Simple TTS function for demo"""
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            
            # Set properties
            engine.setProperty('rate', 180 if language_code == 'en' else 150)
            engine.setProperty('volume', 0.9)
            
            # Try to find appropriate voice
            for voice in voices:
                if language_code == 'ar' and ('arabic' in voice.name.lower() or 'ar-' in voice.id.lower()):
                    engine.setProperty('voice', voice.id)
                    break
                elif language_code == 'en' and ('english' in voice.name.lower() or 'en-' in voice.id.lower()):
                    engine.setProperty('voice', voice.id)
                    break
            
            print(f"🗣️ Speaking: {text}")
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            
        except Exception as e:
            print(f"❌ TTS Error: {e}")
    
    # Demo scenarios
    scenarios = [
        {
            "weather": "Shine",
            "confidence": 95.5,
            "english": "The weather prediction is Shine with 95.5% confidence. Clear skies, great for outdoor activities!",
            "arabic": "التنبؤ بالطقس هو مشمس بثقة 95.5%. سماء صافية، طقس مناسب للنشاطات الخارجية!"
        },
        {
            "weather": "Rain", 
            "confidence": 87.2,
            "english": "The weather prediction is Rain with 87.2% confidence. Rain is expected, grab an umbrella!",
            "arabic": "التنبؤ بالطقس هو ممطر بثقة 87.2%. من المتوقع هطول أمطار، لا تنس المظلة!"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📋 Demo {i}: {scenario['weather']} weather")
        print("-" * 30)
        
        # English announcement
        print("🇺🇸 English announcement:")
        speak_text(scenario['english'], 'en')
        time.sleep(1)
        
        # Arabic announcement  
        print("🇸🇦 Arabic announcement:")
        speak_text(scenario['arabic'], 'ar')
        time.sleep(2)
    
    print("\n✅ Voice demo completed!")
    print("\nℹ️ In the Streamlit app:")
    print("1. Toggle 'Voice Announcements' to enable")
    print("2. Select your preferred language")
    print("3. Upload/capture an image and predict")
    print("4. Listen to the voice announcement!")

if __name__ == "__main__":
    demo_voice_functionality()
