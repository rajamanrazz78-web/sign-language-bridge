import tkinter as tk
from tkinter import ttk
import subprocess, sys, os

BG = "#0d1117"
ACCENT = "#00ffb3"
BTN_BG = "#161b22"

def launch(script):
    subprocess.Popen([sys.executable, os.path.join("src", script)])

root = tk.Tk()
root.title("Sign Language Bridge")
root.geometry("480x420")
root.configure(bg=BG)
root.resizable(False, False)

tk.Label(root, text="🤟 Sign Language Bridge",
         font=("Helvetica", 20, "bold"), bg=BG, fg=ACCENT).pack(pady=30)

tk.Label(root, text="Two-way communication for deaf accessibility",
         font=("Helvetica", 10), bg=BG, fg="#8b949e").pack()

ttk.Separator(root, orient='horizontal').pack(fill='x', pady=20, padx=40)

buttons = [
    ("🎤  Voice → Sign Images",   "speech_to_sign.py",  "Speak and see sign images displayed"),
    ("🖐️  Sign → Voice",          "sign_to_voice.py",   "Sign with your hand, hear it spoken"),
    ("📷  Collect Training Data", "collect_data.py",    "Record gestures for training"),
    ("🤖  Train Model",           "train_model.py",     "Train the gesture classifier"),
]

for label, script, tooltip in buttons:
    frame = tk.Frame(root, bg=BG)
    frame.pack(pady=6)
    btn = tk.Button(frame, text=label, font=("Helvetica", 12),
                    bg=BTN_BG, fg="white", activebackground=ACCENT,
                    activeforeground=BG, relief="flat", padx=20, pady=10,
                    width=32, cursor="hand2",
                    command=lambda s=script: launch(s))
    btn.pack()
    tk.Label(frame, text=tooltip, font=("Helvetica", 8),
             bg=BG, fg="#484f58").pack()

tk.Label(root, text="Press Q in any window to stop",
         font=("Helvetica", 9), bg=BG, fg="#484f58").pack(side="bottom", pady=12)

root.mainloop()