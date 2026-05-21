import tkinter as tk
from tkinter import messagebox
import joblib

model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict_spam():
    email_text = text_input.get("1.0", tk.END).strip()

    if not email_text:
        messagebox.showwarning("Input Error", "Please enter an email message.")
        return

    email_vector = vectorizer.transform([email_text])
    prediction = model.predict(email_vector)[0]
    probability = model.predict_proba(email_vector)[0]

    if prediction == 1:
        result_label.config(
            text=f"Result: Spam\nConfidence: {max(probability)*100:.2f}%",
            fg="red"
        )
    else:
        result_label.config(
            text=f"Result: Not Spam\nConfidence: {max(probability)*100:.2f}%",
            fg="green"
        )

app = tk.Tk()
app.title("Spam Email Detector")
app.geometry("600x450")
app.config(bg="#f4f4f4")

title_label = tk.Label(
    app,
    text="Spam Email Detection",
    font=("Arial", 18, "bold"),
    bg="#f4f4f4"
)
title_label.pack(pady=10)

instruction_label = tk.Label(
    app,
    text="Enter email text below:",
    font=("Arial", 12),
    bg="#f4f4f4"
)
instruction_label.pack()

text_input = tk.Text(app, height=12, width=60, font=("Arial", 11))
text_input.pack(pady=10)

check_button = tk.Button(
    app,
    text="Check Email",
    font=("Arial", 12, "bold"),
    bg="#007acc",
    fg="white",
    command=predict_spam
)
check_button.pack(pady=10)

result_label = tk.Label(
    app,
    text="Result will appear here",
    font=("Arial", 14, "bold"),
    bg="#f4f4f4"
)
result_label.pack(pady=20)

app.mainloop()