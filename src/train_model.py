import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

GESTURES = ['A','B','C','D','E','F','G','H','I','J',
            'K','L','M','N','O','P','Q','R','S','T',
            'U','V','W','X','Y','Z']
DATA_DIR = 'data/processed'

def load_data():
    X, y = [], []
    for gesture in GESTURES:
        x_path = f"{DATA_DIR}/{gesture}_X.npy"
        y_path = f"{DATA_DIR}/{gesture}_y.npy"
        if os.path.exists(x_path):
            X.append(np.load(x_path))
            y.append(np.load(y_path))
    if not X:
        print("❌ No data found in data/processed/")
        print("   Run collect_data.py first!")
        return None, None
    return np.vstack(X), np.concatenate(y)

def train():
    print("📦 Loading data...")
    X, y = load_data()
    if X is None:
        return

    print(f"   Total samples: {len(X)}, Classes: {np.unique(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🤖 Training Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(classification_report(y_test, y_pred))

    os.makedirs('models', exist_ok=True)
    with open('models/gesture_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("💾 Model saved to models/gesture_model.pkl")

if __name__ == "__main__":
    train()