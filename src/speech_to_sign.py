import speech_recognition as sr
import cv2
import os
import time
from PIL import Image, ImageDraw, ImageFont
import numpy as np

SIGNS_DIR = "assets/signs"          # put your ASL letter/word images here
DISPLAY_TIME = 1.2                   # seconds per sign image

def text_to_sign_images(text: str):
    """Yield sign images for each letter/word in text."""
    words = text.upper().split()
    for word in words:
        word_img_path = os.path.join(SIGNS_DIR, f"{word}.jpg")
        if os.path.exists(word_img_path):
            # Whole-word sign exists (e.g. HELLO.png, THANKYOU.png)
            yield word, cv2.imread(word_img_path)
        else:
            # Fingerspell letter by letter
            for letter in word:
                img_path = os.path.join(SIGNS_DIR, f"{letter}.jpg")
                if os.path.exists(img_path):
                    yield letter, cv2.imread(img_path)
                else:
                    # Generate placeholder if image missing
                    yield letter, make_placeholder(letter)

def make_placeholder(char: str):
    """Create a simple placeholder image for missing signs."""
    img = np.ones((400, 400, 3), dtype=np.uint8) * 30
    cv2.putText(img, char, (130, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 255, 200), 12)
    return img

def listen_and_display():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("🎤 Listening... (say something, Ctrl+C to stop)")
    cv2.namedWindow("Sign Language Output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sign Language Output", 600, 500)

    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)

    while True:
        try:
            with mic as source:
                print("\n🔴 Speak now...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)

            text = recognizer.recognize_google(audio)
            print(f"📝 Recognized: '{text}'")

            # Show recognized text on screen first
            info = np.zeros((100, 600, 3), dtype=np.uint8)
            cv2.putText(info, f"'{text}'", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 180), 2)
            cv2.imshow("Sign Language Output", info)
            cv2.waitKey(1500)

            # Display sign images one by one
            for label, img in text_to_sign_images(text):
                if img is None:
                    continue
                display = cv2.resize(img, (500, 500))
                # Add label bar
                bar = np.zeros((60, 500, 3), dtype=np.uint8)
                cv2.putText(bar, label, (200, 42),
                            cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 255, 255), 2)
                combined = np.vstack([display, bar])
                cv2.imshow("Sign Language Output", combined)
                if cv2.waitKey(int(DISPLAY_TIME * 1000)) & 0xFF == ord('q'):
                    break

        except sr.WaitTimeoutError:
            print("⏳ No speech detected, waiting...")
        except sr.UnknownValueError:
            print("❓ Could not understand audio")
        except KeyboardInterrupt:
            print("\n👋 Stopping.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    listen_and_display()