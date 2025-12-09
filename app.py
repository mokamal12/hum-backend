import os
import json
import tempfile
import math

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

import soundfile as sf
import numpy as np
import speech_recognition as sr
from langdetect import detect, LangDetectException

# --------------------
# CONFIG
# --------------------
AUDD_API_KEY = os.getenv("AUDD_API_KEY", "YOUR_AUDD_KEY_HERE")

app = Flask(__name__)
CORS(app)


# --------------------
# HELPERS
# --------------------
def rms_dbfs(audio_path):
    """
    Compute RMS loudness in dBFS.
    Used to detect if user is whispering / too quiet.
    """
    data, sr_ = sf.read(audio_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)  # convert to mono

    if len(data) == 0:
        return -120.0

    rms = np.sqrt(np.mean(np.square(data)))
    db = 20 * math.log10(rms + 1e-12)
    return db


def call_audd(file_path):
    """Send audio file to Audd.io and return raw JSON."""
    url = "https://api.audd.io/"

    data = {
        "api_token": AUDD_API_KEY,
        "return": "apple_music,spotify,deezer,lyrics"
    }

    with open(file_path, "rb") as f:
        files = {"file": ("audio.wav", f, "audio/wav")}
        resp = requests.post(url, data=data, files=files, timeout=30)

    try:
        return resp.json()
    except Exception:
        return None


def simplify_audd_result(audd_json):
    """Convert Audd result to compact JSON for the frontend."""
    if not audd_json or not audd_json.get("result"):
        return None

    s = audd_json["result"]
    out = {
        "title": s.get("title"),
        "artist": s.get("artist"),
        "album": s.get("album"),
        "timecode": s.get("timecode"),
        "lyrics_snippet": None,
        "cover": None,
        "links": {}
    }

    # Lyrics snippet (Arabic or English) – keep UTF-8
    lyrics = s.get("lyrics")
    if isinstance(lyrics, str) and lyrics.strip():
        out["lyrics_snippet"] = lyrics[:400] + ("…" if len(lyrics) > 400 else "")

    # Apple Music cover
    am = s.get("apple_music") or {}
    am_art = am.get("artwork") or {}
    if am_art.get("url"):
        out["cover"] = am_art["url"]

    # Spotify cover fallback
    sp = s.get("spotify") or {}
    sp_album = sp.get("album") or {}
    sp_images = sp_album.get("images") or []
    if not out["cover"] and sp_images:
        out["cover"] = sp_images[0].get("url")

    # Links
    if sp.get("external_urls", {}).get("spotify"):
        out["links"]["spotify"] = sp["external_urls"]["spotify"]
    if am.get("url"):
        out["links"]["apple_music"] = am["url"]
    dz = s.get("deezer") or {}
    if dz.get("link"):
        out["links"]["deezer"] = dz["link"]

    return out


def transcribe_and_detect_lang(audio_path):
    """
    Try to get lyrics text using Google Speech Recognition
    and detect whether it is Arabic or English.
    """
    recognizer = sr.Recognizer()
    text = ""
    lang_guess = "unknown"

    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
    except Exception:
        return None, "unknown"

    # Try English
    try:
        text = recognizer.recognize_google(audio, language="en-US")
    except Exception:
        text = ""

    # If still empty, try Arabic
    if not text.strip():
        try:
            text = recognizer.recognize_google(audio, language="ar-EG")
        except Exception:
            text = ""

    if not text.strip():
        return None, "unknown"

    # Detect language from text
    try:
        code = detect(text)
        if code.startswith("ar"):
            lang_guess = "arabic"
        elif code.startswith("en"):
            lang_guess = "english"
        else:
            lang_guess = "other"
    except LangDetectException:
        lang_guess = "unknown"

    return text, lang_guess


def search_itunes(query, language, era):
    """
    Use iTunes Search API as a free metadata source.
    We use lyrics snippet or user-entered query text.
    """
    if not query:
        return []

    # Decide country based on language
    if language == "arabic":
        country = "eg"  # Egypt
    else:
        country = "us"

    params = {
        "term": query,
        "entity": "song",
        "limit": 15,
        "country": country
    }

    resp = requests.get("https://itunes.apple.com/search", params=params, timeout=20)
    try:
        data = resp.json()
    except Exception:
        return []

    results = []
    for item in data.get("results", []):
        track_name = item.get("trackName")
        artist = item.get("artistName")
        album = item.get("collectionName")
        cover = item.get("artworkUrl100")
        preview = item.get("previewUrl")
        release_date = item.get("releaseDate", "")

        # Era filter by rough year
        year = None
        if release_date:
            try:
                year = int(release_date[:4])
            except Exception:
                year = None

        if era == "old" and year is not None and year >= 2005:
            continue
        if era == "modern" and year is not None and year < 2005:
            continue

        results.append(
            {
                "title": track_name,
                "artist": artist,
                "album": album,
                "cover": cover,
                "preview_url": preview,
                "release_year": year
            }
        )

    return results


# --------------------
# ROUTES
# --------------------
@app.get("/")
def home():
    return jsonify(
        {
            "status": "online",
            "message": "Hum backend is running",
            "endpoints": ["/identify", "/suggest"],
        }
    )


@app.post("/identify")
def identify():
    """
    Main endpoint:
    1) Check volume – if too quiet → "quiet" status
    2) Try Audd – if match → "match" status
    3) If no match → transcribe + guess language and return narrowing info.
    """
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400

    file_storage = request.files["file"]

    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    file_storage.save(tmp_path)

    try:
        # 1) Volume check
        loudness = rms_dbfs(tmp_path)
        # print("DEBUG loudness dBFS:", loudness)

        # Threshold: very quiet / whisper
        if loudness < -45.0:
            return jsonify(
                {
                    "status": "quiet",
                    "loudness_db": loudness,
                    "message": "Input too quiet",
                }
            )

        # 2) Try Audd
        audd_raw = call_audd(tmp_path)
        simplified = simplify_audd_result(audd_raw)

        if simplified is not None:
            # Song found
            return app.response_class(
                response=json.dumps(
                    {"status": "match", "song": simplified}, ensure_ascii=False
                ),
                status=200,
                mimetype="application/json",
            )

        # 3) No match → try to get transcript & language
        transcript, lang_guess = transcribe_and_detect_lang(tmp_path)

        return app.response_class(
            response=json.dumps(
                {
                    "status": "no_match",
                    "message": "No song match from Audd",
                    "transcript": transcript,
                    "language_guess": lang_guess,
                    "filters": {
                        "languages": ["arabic", "english"],
                        "genders": ["male", "female"],
                        "eras": ["old", "modern"],
                    },
                },
                ensure_ascii=False,
            ),
            status=200,
            mimetype="application/json",
        )

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/suggest")
def suggest():
    """
    Suggest songs based on:
    - transcript text (lyrics),
    - language,
    - era,
    - (gender is just passed through; real gender filtering would need a richer DB)
    """
    data = request.get_json(silent=True) or {}
    transcript = data.get("transcript") or ""
    language = data.get("language") or "english"
    era = data.get("era") or "modern"
    gender = data.get("gender") or "unknown"

    # Use a short query phrase
    query = transcript.strip()
    if len(query.split()) > 8:
        query = " ".join(query.split()[:8])

    songs = search_itunes(query, language, era)

    return app.response_class(
        response=json.dumps(
            {
                "status": "ok",
                "language": language,
                "era": era,
                "gender": gender,
                "query_used": query,
                "results": songs,
            },
            ensure_ascii=False,
        ),
        status=200,
        mimetype="application/json",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
