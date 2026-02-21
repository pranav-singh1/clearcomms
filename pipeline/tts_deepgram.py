import json
import urllib.request
import urllib.error

def deepgram_tts_mp3(text: str, api_key: str, model: str = "aura-asteria-en") -> bytes:
    """
    Deepgram TTS -> returns MP3 bytes.
    If your team uses a different Deepgram endpoint/model name, edit the URL below.
    """
    if not api_key:
        raise ValueError("Missing Deepgram API key")

    url = f"https://api.deepgram.com/v1/speak?model={model}&encoding=mp3"
    payload = json.dumps({"text": text}).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Deepgram HTTPError {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Deepgram URLError: {e}") from e
