import os
import json
import tempfile
import math

from flask import Flask, request, jsonify
from flask_cors import CORS

import requests
import soundfile as sf
import numpy as np

# --------------------
# CONFIG
# --------------------
AUDD_API_KEY = os.getenv("bb413cb2c6c98518d943fb4ed00276e1")

app = Flask(__name__)
CORS(app)


# --------------------
# AUDIO UTIL – LOUDNESS
# --------------------
def rms_dbfs(audio_path):
    """
    Compute loudness in dBFS to detect whisper / too-quiet input.
    """
    data, sr = sf.read(audio_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if len(data) == 0:
        return -120.0

    rms = np.sqrt(np.mean(np.square(data)))
    db = 20 * math.log10(rms + 1e-12)
    return db


# --------------------
# AUDD INTEGRATION
# --------------------
def call_audd(file_path):
    """Send audio file to Audd.io and return JSON."""
    url = "https://api.audd.io/"
    data = {
        "api_token": AUDD_API_KEY,
        "return": "apple_music,spotify,deezer,lyrics",
    }

    with open(file_path, "rb") as f:
        files = {"file": ("audio.wav", f, "audio/wav")}
        resp = requests.post(url, data=data, files=files, timeout=30)

    try:
        return resp.json()
    except Exception:
        return None


def simplify_audd_result(audd_json):
    """Convert Audd result to a compact object for the frontend."""
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
        "links": {},
    }

    # Lyrics snippet – keep UTF-8
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


# --------------------
# iTunes SEARCH + SCORING FOR LYRICS
# --------------------
def jaccard_similarity(a, b):
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def search_itunes_by_lyrics(lyrics, language, era):
    """
    Use lyrics text as the query to iTunes Search API.
    language: 'arabic' or 'english'
    era: 'old' or 'modern'
    """
    if not lyrics:
        return []

    # keep only first ~8–10 words to avoid long queries
    tokens = lyrics.strip().split()
    if len(tokens) > 10:
        tokens = tokens[:10]
    query = " ".join(tokens)

    # country based on language
    if language == "arabic":
        country = "eg"  # Egypt store
    else:
        country = "us"

    params = {
        "term": query,
        "entity": "song",
        "limit": 25,
        "country": country,
        "media": "music",
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
        duration_ms = item.get("trackTimeMillis")

        year = None
        if release_date:
            try:
                year = int(release_date[:4])
            except Exception:
                year = None

        # Era filter
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
                "release_year": year,
                "duration_ms": duration_ms,
            }
        )

    return results


def score_song(song, lyrics_tokens):
    """
    Score by overlap between lyrics tokens and (title + artist).
    """
    title = (song.get("title") or "").lower()
    artist = (song.get("artist") or "").lower()
    all_text = (title + " " + artist).split()
    return float(jaccard_similarity(lyrics_tokens, all_text))


# --------------------
# ROUTES
# --------------------
@app.get("/")
def home():
    return jsonify(
        {
            "status": "online",
            "message": "Hum backend is running",
            "endpoints": ["/identify", "/lyrics"],
        }
    )


@app.post("/identify")
def identify():
    """
    Audio-based identification (humming or singing).
    Uses Audd. This is used when browser STT fails (humming).
    """
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400

    file_storage = request.files["file"]

    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    file_storage.save(tmp_path)

    try:
        loudness = rms_dbfs(tmp_path)
        if loudness < -45.0:
            return jsonify(
                {
                    "status": "quiet",
                    "loudness_db": loudness,
                    "message": "Input too quiet",
                }
            )

        audd_raw = call_audd(tmp_path)
        simplified = simplify_audd_result(audd_raw)

        if simplified is not None:
            return app.response_class(
                response=json.dumps(
                    {
                        "status": "match",
                        "song": simplified,
                    },
                    ensure_ascii=False,
                ),
                status=200,
                mimetype="application/json",
            )
        else:
            return app.response_class(
                response=json.dumps(
                    {
                        "status": "no_match",
                        "message": "No exact match from Audd",
                    },
                    ensure_ascii=False,
                ),
                status=200,
                mimetype="application/json",
            )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/lyrics")
def lyrics_search():
    """
    Lyrics-based identification.
    This is called from the frontend after browser speech recognition
    (Arabic/English) gets the text.
    """
    data = request.get_json(silent=True) or {}
    lyrics = data.get("lyrics") or ""
    language = data.get("language") or "english"  # 'arabic' or 'english'
    era = data.get("era") or "modern"             # 'old' or 'modern'
    gender = data.get("gender") or "unknown"      # not used yet, kept for future

    lyrics = lyrics.strip()
    if not lyrics:
        return jsonify({"status": "error", "message": "Empty lyrics"}), 400

    # basic tokenization
    raw_tokens = lyrics.lower().split()
    lyrics_tokens = [t for t in raw_tokens if len(t) > 2]

    songs = search_itunes_by_lyrics(lyrics, language, era)

    scored = []
    for s in songs:
        sc = score_song(s, lyrics_tokens)
        ss = dict(s)
        ss["score"] = sc
        scored.append(ss)

    scored.sort(key=lambda x: x["score"], reverse=True)
    best_match = scored[0] if scored else None

    return app.response_class(
        response=json.dumps(
            {
                "status": "ok",
                "language": language,
                "era": era,
                "gender": gender,
                "lyrics_used": lyrics,
                "best_match": best_match,
                "results": scored,
            },
            ensure_ascii=False,
        ),
        status=200,
        mimetype="application/json",
    )


if __name__ == "__main__":
    # For local testing; Railway uses gunicorn
    app.run(host="0.0.0.0", port=5000, debug=True)
