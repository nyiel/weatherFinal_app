#!/usr/bin/env python3
"""Test script for voice functionality"""

from model_utils import text_to_speech, get_voice_announcement
import time

def test_voice():
    print("Testing voice functionality...")
    
    # Test English announcement
    print("Testing English...")
    english_text = get_voice_announcement('Shine', 'English', 95.5)
    print(f"English text: {english_text}")
    text_to_speech(english_text, 'en')
    
    time.sleep(3)  # Wait a bit
    
    # Test Arabic announcement
    print("Testing Arabic...")
    arabic_text = get_voice_announcement('Rain', 'العربية', 87.2)
    print(f"Arabic text: {arabic_text}")
    text_to_speech(arabic_text, 'ar')
    
    print("Voice test completed!")

if __name__ == "__main__":
    test_voice()
