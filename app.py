import os
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

# --------------------
# CONFIG
# --------------------
AUDD_API_KEY = os.getenv("AUDD_API_KEY", "YOUR_AUDD_KEY_HERE")  # <-- change this if you want

app = Flask(__name__)
CORS(app)


def call_audd(file_storage):
    """Send uploaded audio file to Audd.io and return JSON."""
    url = "https://api.audd.io/"
    data = {
        "api_token": AUDD_API_KEY,
        "return": "apple_music,spotify,deezer,lyrics"
    }

    files = {
        "file": (file_storage.filename, file_storage.stream, file_storage.mimetype)
    }

    resp = requests.post(url, data=data, files=files, timeout=30)
    try:
        return resp.json()
    except Exception:
        return None


def simplify_result(audd_json):
    """Convert Audd result to a clean, small JSON object for the frontend."""
    if not audd_json or not audd_json.get("result"):
        return {"found": False}

    s = audd_json["result"]
    simplified = {
        "found": True,
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
        simplified["lyrics_snippet"] = lyrics[:400] + ("…" if len(lyrics) > 400 else "")

    # Try Apple Music cover first
    am = s.get("apple_music") or {}
    am_art = am.get("artwork") or {}
    if am_art.get("url"):
        simplified["cover"] = am_art["url"]

    # Then Spotify cover as fallback
    sp = s.get("spotify") or {}
    sp_album = sp.get("album") or {}
    sp_images = sp_album.get("images") or []
    if not simplified["cover"] and sp_images:
        simplified["cover"] = sp_images[0].get("url")

    # Streaming links
    if sp.get("external_urls", {}).get("spotify"):
        simplified["links"]["spotify"] = sp["external_urls"]["spotify"]
    if am.get("url"):
        simplified["links"]["apple_music"] = am["url"]
    dz = s.get("deezer") or {}
    if dz.get("link"):
        simplified["links"]["deezer"] = dz["link"]

    return simplified


@app.route("/identify", methods=["POST"])
def identify():
    """Main endpoint: receives audio, sends to Audd, returns simplified JSON."""
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400

    audio_file = request.files["file"]

    audd_raw = call_audd(audio_file)
    simplified = simplify_result(audd_raw)

    # Ensure Arabic is kept correctly using UTF-8 JSON
    return app.response_class(
        response=json.dumps(simplified, ensure_ascii=False),
        status=200,
        mimetype="application/json"
    )


if __name__ == "__main__":
    # For local testing; on Render we'll use gunicorn
    app.run(host="0.0.0.0", port=5000, debug=True)
