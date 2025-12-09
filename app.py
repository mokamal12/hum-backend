import os
import io
import json
import librosa
import numpy as np
import soundfile as sf
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import speech_recognition as sr
from langdetect import detect

app = Flask(__name__)
CORS(app)

# --------------------
# CONFIG
# --------------------
AUDD_API_KEY = os.getenv("AUDD_API_KEY", "948e2444a74bf363ab42c4022af26824")


# --------------------
# HOME ROUTE
# --------------------
@app.route("/")
def home():
    return {
        "status": "online",
        "message": "Hum backend is running",
        "endpoints": ["/identify"]
    }


# --------------------
# VOCALS DETECTION (lyrics vs humming)
# --------------------
def detect_if_vocals(audio_data, sr_rate):
    try:
        S = np.abs(librosa.stft(audio_data))
        centroid = librosa.feature.spectral_centroid(S=S)[0]
        return np.mean(centroid) > 1500
    except:
        return False


# --------------------
# SPEECH TO TEXT
# --------------------
def extract_text(file_bytes):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(file_bytes)) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio, language="ar-EG")
        return text
    except:
        return None


# --------------------
# LANGUAGE DETECTION
# --------------------
def detect_language(text):
    try:
        lang = detect(text)
        return "arabic" if lang.startswith("ar") else "english"
    except:
        return "unknown"


# --------------------
# HUMMING IDENTIFICATION (Audd API)
# --------------------
def identify_humming(file_bytes):
    url = "https://api.audd.io/"
    files = {"file": ("audio.wav", file_bytes, "audio/wav")}
    data = {
        "api_token": AUDD_API_KEY,
        "return": "spotify,apple_music,deezer"
    }
    return requests.post(url, files=files, data=data).json()


# --------------------
# MAIN IDENTIFY ENDPOINT
# --------------------
@app.route("/identify", methods=["POST"])
def identify():
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400

    uploaded = request.files["file"]
    file_bytes = uploaded.read()

    try:
        audio_data, sr_rate = librosa.load(io.BytesIO(file_bytes), sr=16000)
    except:
        return jsonify({"error": "invalid audio"}), 400

    # Detect humming or vocals
    vocals = detect_if_vocals(audio_data, sr_rate)

    # ----- LYRICS MODE -----
    if vocals:
        text = extract_text(file_bytes)
        lang = detect_language(text) if text else "unknown"
        return jsonify({
            "mode": "lyrics",
            "text": text,
            "language": lang
        })

    # ----- HUMMING MODE -----
    result = identify_humming(file_bytes)
    return jsonify({
        "mode": "humming",
        "result": result.get("result")
    })


# --------------------
# RAILWAY SERVER MODE
# --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
