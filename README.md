# Sign Language Bridge

A two-way communication system that bridges the gap between deaf and hearing people using computer vision and machine learning.

---

## 🎯 What This Project Does

### 🎤 Hearing → Deaf

Speak into microphone

↓

Speech converted to text

↓

Text converted to Sign Language images

↓

Deaf person sees the signs on screen

---

## ✨ Features

- 🎤 **Voice → Sign Language** — Speak and see ASL signs displayed
- 🖐️ **Sign → Voice** — Show hand signs, computer speaks it out
- 🔤 **A to Z Support** — Full ASL alphabet recognition
- 📷 **Real-time Detection** — Live webcam hand tracking
- 🤖 **Machine Learning** — Random Forest classifier with 99% accuracy

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Core language |
| OpenCV | Camera and image processing |
| MediaPipe | Hand landmark detection |
| scikit-learn | Gesture classification model |
| pyttsx3 | Text to speech |
| SpeechRecognition | Voice to text |
| NumPy | Data processing |

---

## 📁 Project Structure

sign_language_bridge/

├── src/

│   ├── collect_data.py      # Record hand gesture training data

│   ├── train_model.py       # Train the ML classifier

│   ├── speech_to_sign.py    # Voice → Sign images pipeline

│   ├── sign_to_voice.py     # Sign → Voice pipeline

│   └── main_app.py          # Main GUI launcher

├── assets/

│   └── signs/               # ASL letter images (A.jpg - Z.jpg)

├── data/

│   └── processed/           # Training data - auto generated

├── models/

│   └── gesture_model.pkl    # Trained model - auto generated

├── .gitignore

├── requirements.txt

└── README.md


---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/TUMHARA_USERNAME/sign-language-bridge.git
cd sign-language-bridge
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add ASL Sign Images

---

## 🚀 Usage

### Step 1 — Collect Training Data
```bash
python src/collect_data.py
```
- Webcam opens
- Show each hand sign
- Press SPACE to record
- 200 samples per gesture

### Step 2 — Train the Model
```bash
python src/train_model.py
```
- Trains Random Forest classifier
- Saves model to models/gesture_model.pkl
- Achieves ~99% accuracy

### Step 3 — Launch the App
```bash
python src/main_app.py
```

---

## 🎮 Controls

### Sign → Voice Mode
| Key | Action |
|---|---|
| Hold sign 1.5s | Add letter |
| S | Speak full sentence |
| C | Clear sentence |
| Q | Quit |

### Voice → Sign Mode

---

## 📈 Improving Accuracy

```bash
# Collect more data
python src/collect_data.py

# Retrain model
python src/train_model.py

# Test again
python src/sign_to_voice.py
```

---

## 🙏 Accessibility Impact

This project aims to:
- Help deaf people communicate with hearing people
- Remove communication barriers in hospitals, schools, offices
- Make technology more inclusive and accessible

---

## 👨‍💻 Developer

**Raj**
- GitHub: [@rajamanrazz78-web](https://github.com/rajamanrazz78-web)
- Linkedin - www.linkedin.com/in/raj-aman-2560ab275

---

## 📄 License

MIT License — feel free to use and improve!

