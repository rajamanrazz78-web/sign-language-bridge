import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import time
from collections import deque, Counter

# Model load karo
with open('models/gesture_model.pkl', 'rb') as f:
    model = pickle.load(f)

# TTS
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

# Mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
    max_num_hands=1
)

pred_buffer = deque(maxlen=7)
sentence = []
last_spoken = None
hold_start = None
HOLD_SECONDS = 1.5

# NAYA - yeh replace karo
def speak(text):
    print(f"🔊 Bol raha: {text}")
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()
    del engine

cap = cv2.VideoCapture(0)
print("✅ Chalu hai! Haath dikhao.")
print("S = bolo | C = clear | Q = band karo")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    pred = None
    conf = 0.0

    if results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hl, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            features = np.array([[lm.x, lm.y, lm.z]
                                  for lm in hl.landmark]).flatten().reshape(1, -1)
            conf = model.predict_proba(features).max()
            pred = model.predict(features)[0]
            pred_buffer.append(pred)
            print(f"Detecting: {pred} ({conf:.0%})")  # terminal mein dikhega

    # Stable prediction
    stable = None
    if len(pred_buffer) == 7:
        most_common, freq = Counter(pred_buffer).most_common(1)[0]
        if freq >= 5:
            stable = most_common

    # Hold to confirm
    if stable and stable != last_spoken:
        if hold_start is None:
            hold_start = time.time()
        elapsed = time.time() - hold_start

        # Progress bar
        bar = int((elapsed / HOLD_SECONDS) * frame.shape[1])
        cv2.rectangle(frame, (0, 0), (bar, 10), (0, 255, 180), -1)

        if elapsed >= HOLD_SECONDS:
            sentence.append(stable)
            last_spoken = stable
            hold_start = None
            pred_buffer.clear()
            speak(stable)
    else:
        hold_start = None
        if not stable:
            last_spoken = None

    # UI
    cv2.rectangle(frame, (0, 12), (frame.shape[1], 70), (0, 0, 0), -1)
    if stable:
        txt = f"Sign: {stable} ({conf:.0%})"
        col = (0, 255, 180)
    elif results.multi_hand_landmarks:
        txt = f"Dekh raha: {pred} ({conf:.0%})" if pred else "Analyzing..."
        col = (0, 200, 255)
    else:
        txt = "Haath dikhao!"
        col = (0, 100, 255)

    cv2.putText(frame, txt, (15, 55),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, col, 2)

    # Sentence bar
    cv2.rectangle(frame, (0, frame.shape[0]-45),
                  (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.putText(frame, ' '.join(sentence) if sentence else "...",
                (15, frame.shape[0]-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 220, 100), 2)

    cv2.imshow("Sign to Voice", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and sentence:
        speak(' '.join(sentence))
    elif key == ord('c'):
        sentence.clear()
        last_spoken = None
        print("Clear!")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()