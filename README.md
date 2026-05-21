# Spam Email Detector GUI

A Python machine learning project that detects whether a message is spam or ham using a trained text classification model and a Tkinter GUI.

## Project Structure

```text
spam-email-detector/
├── data/
│   └── spam.csv
├── models/
├── src/
│   ├── train_model.py
│   └── gui_app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Train the Model

```bash
python src/train_model.py
```

## Run the GUI

```bash
python src/gui_app.py
```