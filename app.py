import os
import json
import math
import tempfile
import urllib.parse

import requests
import soundfile as sf
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup

# ==============================
# CONFIG
# ==============================
AUDD_API_KEY = os.getenv("bb413cb2c6c98518d943fb4ed00276e1")

app = Flask(__name__)
CORS(app)


# ---------- Utility: loudness check ----------
def rms_dbfs(audio_path: str) -> float:
    """Compute loudness in dBFS to detect whisper / too-quiet input."""
    data, sr = sf.read(audio_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if len(data) == 0:
        return -120.0

    rms = np.sqrt(np.mean(np.square(data)))
    db = 20 * math.log10(rms + 1e-12)
    return db


# ---------- Audd.io recognition ----------
def call_audd(file_path: str):
    """
    Send audio file to Audd.io.
    This handles:
      - Played song (like Shazam)
      - Singing with lyrics
      - Humming (if Audd can catch it)
    """
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
        "lyrics_full": None,
        "cover": None,
        "links": {},
    }

    # Lyrics (UTF-8 safe)
    lyrics = s.get("lyrics")
    if isinstance(lyrics, str) and lyrics.strip():
        out["lyrics_full"] = lyrics

    # Apple Music artwork
    am = s.get("apple_music") or {}
    am_art = am.get("artwork") or {}
    if am_art.get("url"):
        out["cover"] = am_art["url"]

    # Spotify artwork fallback
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


# ---------- Genius helpers (optional, best effort) ----------
def genius_search_song(query: str):
    """
    Use Genius search page to find a song URL.
    This is not an official API but usually works.
    """
    try:
        q = urllib.parse.quote(query)
        url = f"https://genius.com/api/search/multi?per_page=5&q={q}"
        resp = requests.get(url, timeout=15)
        data = resp.json()
    except Exception:
        return None

    sections = data.get("response", {}).get("sections", [])
    for section in sections:
        if section.get("type") == "song":
            hits = section.get("hits", [])
            if not hits:
                continue
            song = hits[0].get("result", {})
            path = song.get("path")
            title = song.get("title")
            artist = song.get("primary_artist", {}).get("name")
            if path:
                return {
                    "url": "https://genius.com" + path,
                    "title": title,
                    "artist": artist,
                }

    return None


def genius_fetch_lyrics(url: str) -> str | None:
    """
    Scrape full lyrics text from a Genius song page.
    """
    try:
        resp = requests.get(url, timeout=20)
        html = resp.text
        soup = BeautifulSoup(html, "lxml")
        containers = soup.find_all("div", attrs={"data-lyrics-container": "true"})
        if not containers:
            return None
        parts = []
        for c in containers:
            parts.append(c.get_text(separator="\n"))
        full = "\n".join(parts)
        return full.strip()
    except Exception:
        return None


def extract_next_lines(full_lyrics: str, max_lines: int = 4) -> list[str]:
    """
    Very simple: return first N non-empty lines after the first stanza.
    This is a placeholder for "next 4 bars".
    """
    if not full_lyrics:
        return []

    raw_lines = [line.strip() for line in full_lyrics.splitlines()]
    lines = [ln for ln in raw_lines if ln]

    # Just take first N non-empty lines
    return lines[:max_lines]


def build_fun_fact_from_song(song: dict) -> str | None:
    if not song:
        return None
    pieces = []
    title = song.get("title") or ""
    artist = song.get("artist") or ""
    if title and artist:
        pieces.append(f"This match is \"{title}\" by {artist}.")
    elif title:
        pieces.append(f"This track is titled \"{title}\".")
    elif artist:
        pieces.append(f"This track is performed by {artist}.")
    if not pieces:
        return None
    return " ".join(pieces)


# ---------- Simple iTunes lyrics-based search ----------
def search_itunes_by_lyrics(lyrics: str, language: str):
    """
    Use iTunes as a free DB. We query by lyrics snippet.
    """
    if not lyrics:
        return []

    tokens = lyrics.strip().split()
    if len(tokens) > 10:
        tokens = tokens[:10]
    query = " ".join(tokens)

    # Rough language mapping
    country = "eg" if language == "arabic" else "us"

    params = {
        "term": query,
        "entity": "song",
        "limit": 20,
        "country": country,
        "media": "music",
    }

    try:
        resp = requests.get("https://itunes.apple.com/search", params=params, timeout=20)
        data = resp.json()
    except Exception:
        return []

    results = []
    for item in data.get("results", []):
        results.append(
            {
                "title": item.get("trackName"),
                "artist": item.get("artistName"),
                "album": item.get("collectionName"),
                "cover": item.get("artworkUrl100"),
                "preview_url": item.get("previewUrl"),
                "release_year": int(item["releaseDate"][:4]) if item.get("releaseDate") else None,
                "duration_ms": item.get("trackTimeMillis"),
            }
        )
    return results


# ==============================
# ROUTES
# ==============================

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
    Raw audio → Audd (played song, singing, humming).
    Used when:
      - Lyrics recognition on frontend failed
      - Or user just hummed
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
            # Too quiet → whisper
            return jsonify(
                {
                    "status": "quiet",
                    "loudness_db": loudness,
                    "message": "Input too quiet",
                }
            )

        audd_raw = call_audd(tmp_path)
        simplified = simplify_audd_result(audd_raw)

        if simplified is None:
            return jsonify(
                {
                    "status": "no_match",
                    "message": "No exact match from Audd",
                }
            )

        # Try to get full lyrics from Genius, if we know title+artist
        genius_next_lines = []
        if simplified["title"] and simplified["artist"]:
            query = f'{simplified["title"]} {simplified["artist"]}'
            g_song = genius_search_song(query)
            if g_song and g_song.get("url"):
                full_lyrics = genius_fetch_lyrics(g_song["url"])
                genius_next_lines = extract_next_lines(full_lyrics, max_lines=4)

        # Fallback: use Audd lyrics if Genius fails
        if not genius_next_lines and simplified["lyrics_full"]:
            raw_lines = [ln.strip() for ln in simplified["lyrics_full"].splitlines()]
            genius_next_lines = [ln for ln in raw_lines if ln][:4]

        fun_fact = build_fun_fact_from_song(simplified)

        payload = {
            "status": "match",
            "song": {
                "title": simplified["title"],
                "artist": simplified["artist"],
                "album": simplified["album"],
                "cover": simplified["cover"],
                "links": simplified["links"],
            },
            "next_lines": genius_next_lines,
            "fun_fact": fun_fact,
        }

        return app.response_class(
            response=json.dumps(payload, ensure_ascii=False),
            status=200,
            mimetype="application/json",
        )

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/lyrics")
def lyrics_route():
    """
    Lyrics text → best guess (lyrics-first mode).
    Frontend sends:
      - lyrics (transcript from browser speech)
      - language (arabic/english/auto)
      - era, gender (for future ranking refinement, currently informational)
    """
    data = request.get_json(silent=True) or {}
    lyrics = (data.get("lyrics") or "").strip()
    language = data.get("language") or "english"
    era = data.get("era") or "modern"
    gender = data.get("gender") or "unknown"

    if not lyrics:
        return jsonify({"status": "error", "message": "Empty lyrics"}), 400

    # 1) Try Genius search for the lyrics
    g_song = genius_search_song(lyrics)
    next_lines = []
    song_meta = None

    if g_song and g_song.get("url"):
        full_lyrics = genius_fetch_lyrics(g_song["url"])
        next_lines = extract_next_lines(full_lyrics, max_lines=4)
        song_meta = {
            "title": g_song.get("title"),
            "artist": g_song.get("artist"),
            "album": None,
            "cover": None,
            "links": {},
        }

    # 2) iTunes for popularity & extra suggestions
    itunes_results = search_itunes_by_lyrics(lyrics, language)

    best_match = None
    if itunes_results:
        best_match = itunes_results[0]
        if not song_meta:
            song_meta = {
                "title": best_match.get("title"),
                "artist": best_match.get("artist"),
                "album": best_match.get("album"),
                "cover": best_match.get("cover"),
                "links": {},
            }
        else:
            if not song_meta.get("cover") and best_match.get("cover"):
                song_meta["cover"] = best_match["cover"]

    if not song_meta:
        return jsonify(
            {
                "status": "no_match",
                "message": "No clear song matched these lyrics.",
                "results": itunes_results,
            }
        )

    fun_fact = build_fun_fact_from_song(song_meta)

    payload = {
        "status": "ok",
        "language": language,
        "era": era,
        "gender": gender,
        "lyrics_used": lyrics,
        "song": song_meta,
        "next_lines": next_lines,
        "fun_fact": fun_fact,
        "results": itunes_results,
    }

    return app.response_class(
        response=json.dumps(payload, ensure_ascii=False),
        status=200,
        mimetype="application/json",
    )


if __name__ == "__main__":
    # Local testing; Railway uses gunicorn
    app.run(host="0.0.0.0", port=5000, debug=True)
