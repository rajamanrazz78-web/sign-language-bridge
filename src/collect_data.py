import cv2
import mediapipe as mp
import numpy as np
import os
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

GESTURES = ['U', 'V', 'W', 'X', 'Y', 'Z']
SAMPLES_PER_GESTURE = 200          
DATA_DIR = 'data/processed'
os.makedirs(DATA_DIR, exist_ok=True)

def extract_landmarks(hand_landmarks):
    return np.array([[lm.x, lm.y, lm.z]
                     for lm in hand_landmarks.landmark]).flatten()

def collect():
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7
    )

    for gesture in GESTURES:
        print(f"\n{'='*40}")
        print(f"📷 GESTURE: '{gesture}'")
        print(f"{'='*40}")
        print(f"Sahi sign banao aur SPACE dabaao")

        # Countdown before recording
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            # Show which gesture
            cv2.rectangle(frame, (0,0), (640, 80), (0,0,0), -1)
            cv2.putText(frame, f"Banao: '{gesture}' sign",
                       (10, 35), cv2.FONT_HERSHEY_DUPLEX,
                       1.2, (0, 255, 0), 2)
            cv2.putText(frame, "Sign banao phir SPACE dabaao",
                       (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (200, 200, 200), 1)

            # Show hand detection live
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                for hl in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame, "✓ Hand is visible!",
                           (10, frame.shape[0]-20),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "✗ Hand is not visible",
                           (10, frame.shape[0]-20),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.8, (0, 0, 255), 2)

            cv2.imshow("Data Collection", frame)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break

        # 3 second countdown
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.rectangle(frame, (0,0), (640, 80), (0,0,0), -1)
            cv2.putText(frame, f"Recording '{gesture}' in {i}...",
                       (10, 55), cv2.FONT_HERSHEY_DUPLEX,
                       1.2, (0, 200, 255), 2)
            cv2.imshow("Data Collection", frame)
            cv2.waitKey(1000)

        # Collect samples
        samples, labels = [], []
        count = 0

        while count < SAMPLES_PER_GESTURE:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hl in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hl, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0,255,0), thickness=2),
                        mp_draw.DrawingSpec(color=(255,0,0), thickness=2)
                    )
                    features = extract_landmarks(hl)
                    samples.append(features)
                    labels.append(gesture)
                    count += 1

            # Progress bar
            progress = int((count / SAMPLES_PER_GESTURE) * 600)
            cv2.rectangle(frame, (0, frame.shape[0]-15),
                         (progress, frame.shape[0]), (0, 255, 180), -1)

            cv2.rectangle(frame, (0,0), (640, 60), (0,0,0), -1)
            cv2.putText(frame, f"'{gesture}' record ho raha: {count}/{SAMPLES_PER_GESTURE}",
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                       1.0, (0, 255, 255), 2)
            cv2.imshow("Data Collection", frame)
            cv2.waitKey(1)

        np.save(f"{DATA_DIR}/{gesture}_X.npy", np.array(samples))
        np.save(f"{DATA_DIR}/{gesture}_y.npy", np.array(labels))
        print(f"✅ '{gesture}' ke {SAMPLES_PER_GESTURE} samples save ho gaye!")

    cap.release()
    cv2.destroyAllWindows()
    print("\n🎉 all data is collected")
    print("run: python src/train_model.py")

if __name__ == "__main__":
    collect()