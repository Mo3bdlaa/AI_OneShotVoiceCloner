"""Tests for the HTTP API.

The server is driven through a real socket rather than by calling handler methods
directly, so the multipart parsing, status codes and content types are all
covered -- those are exactly the parts that a refactor breaks silently.
"""

from __future__ import annotations

import io
import json
import threading
import urllib.error
import urllib.request
import uuid
from http.server import ThreadingHTTPServer

import pytest

from voxprint.pipeline import VoiceLab
from voxprint.server import PAGE, _Api, _Handler


@pytest.fixture
def server(tmp_path, clips):
    lab = VoiceLab(tmp_path / "voices", require_consent=True)
    handler = type("BoundHandler", (_Handler,), {"api": _Api(lab)})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", clips
    httpd.shutdown()
    httpd.server_close()


def _multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    buf = io.BytesIO()
    for name, value in fields.items():
        buf.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n".encode())
        buf.write(str(value).encode("utf-8") + b"\r\n")
    for name, (filename, data) in files.items():
        buf.write(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; "
            f"filename=\"{filename}\"\r\nContent-Type: audio/wav\r\n\r\n".encode()
        )
        buf.write(data + b"\r\n")
    buf.write(f"--{boundary}--\r\n".encode())
    return buf.getvalue(), f"multipart/form-data; boundary={boundary}"


def post(base, path, fields=None, files=None):
    body, content_type = _multipart(fields or {}, files or {})
    request = urllib.request.Request(base + path, data=body, headers={"Content-Type": content_type})
    with urllib.request.urlopen(request) as response:
        return response.status, response.read(), dict(response.headers)


def get_json(base, path):
    with urllib.request.urlopen(base + path) as response:
        return json.loads(response.read())


def audio_bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


def test_index_serves_the_page(server):
    base, _ = server
    with urllib.request.urlopen(base + "/") as response:
        body = response.read().decode()
    assert response.status == 200
    assert "<title>voxprint</title>" in body
    assert body == PAGE


def test_status_reports_an_empty_gallery(server):
    base, _ = server
    status = get_json(base, "/api/status")
    assert status["speakers"] == 0
    assert status["encoder"]["name"] == "dsp"


def test_enroll_then_identify(server):
    base, clips = server
    status, body, _ = post(
        base, "/api/enroll",
        {"speaker_id": "omar", "consent": "agreed for testing", "granted_by": "fixture"},
        {"audio": ("a.wav", audio_bytes(clips["omar"][0]))},
    )
    assert status == 200
    assert json.loads(body)["speaker_id"] == "omar"

    post(base, "/api/enroll",
         {"speaker_id": "hana", "consent": "agreed for testing"},
         {"audio": ("b.wav", audio_bytes(clips["hana"][0]))})
    post(base, "/api/calibrate")

    _, body, _ = post(base, "/api/identify", files={"audio": ("q.wav", audio_bytes(clips["omar"][3]))})
    result = json.loads(body)
    assert result["candidates"][0]["speaker_id"] == "omar"


def test_enroll_without_consent_is_rejected(server):
    base, clips = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        post(base, "/api/enroll", {"speaker_id": "omar", "consent": "  "},
             {"audio": ("a.wav", audio_bytes(clips["omar"][0]))})
    assert exc.value.code == 400
    assert "consent" in json.loads(exc.value.read())["error"]


def test_missing_audio_is_a_client_error(server):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        post(base, "/api/identify")
    assert exc.value.code == 400
    assert "audio" in json.loads(exc.value.read())["error"]


def test_speakers_lists_consent_state(server):
    base, clips = server
    post(base, "/api/enroll", {"speaker_id": "omar", "consent": "agreed"},
         {"audio": ("a.wav", audio_bytes(clips["omar"][0]))})
    speakers = get_json(base, "/api/speakers")
    assert speakers[0]["speaker_id"] == "omar"
    assert speakers[0]["has_consent"] is True


def test_convert_returns_audio_and_a_measurement(server):
    base, clips = server
    for name in ("omar", "hana"):
        post(base, "/api/enroll", {"speaker_id": name, "consent": "agreed"},
             {"audio": ("a.wav", audio_bytes(clips[name][0]))})
    post(base, "/api/calibrate")

    status, body, headers = post(
        base, "/api/convert", {"speaker_id": "hana"},
        {"audio": ("s.wav", audio_bytes(clips["omar"][0]))},
    )
    assert status == 200
    assert headers["Content-Type"] == "audio/wav"
    assert body.startswith(b"RIFF")
    info = json.loads(headers["X-Voxprint-Info"])
    assert "similarity_to_target" in info


def test_watermark_endpoint_detects_generated_audio(server):
    base, clips = server
    for name in ("omar", "hana"):
        post(base, "/api/enroll", {"speaker_id": name, "consent": "agreed"},
             {"audio": ("a.wav", audio_bytes(clips[name][0]))})
    _, converted, _ = post(base, "/api/convert", {"speaker_id": "hana"},
                           {"audio": ("s.wav", audio_bytes(clips["omar"][0]))})

    _, body, _ = post(base, "/api/watermark", files={"audio": ("c.wav", converted)})
    assert json.loads(body)["present"] is True

    _, body, _ = post(base, "/api/watermark", files={"audio": ("o.wav", audio_bytes(clips["omar"][0]))})
    assert json.loads(body)["present"] is False


def test_remove_deletes_a_speaker(server):
    base, clips = server
    post(base, "/api/enroll", {"speaker_id": "omar", "consent": "agreed"},
         {"audio": ("a.wav", audio_bytes(clips["omar"][0]))})
    post(base, "/api/remove", {"speaker_id": "omar"})
    assert get_json(base, "/api/speakers") == []


def test_unknown_route_is_404(server):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(base + "/api/nope")
    assert exc.value.code == 404


def test_large_upload_with_expect_100_continue(server):
    """curl sends Expect: 100-continue for any body over ~1 KB.

    Under the HTTP/1.0 default of BaseHTTPRequestHandler such a request never
    reaches the handler at all, so every file upload failed while bodyless POSTs
    and GETs worked. This pins the fix.
    """
    import http.client

    base, clips = server
    host = base.removeprefix("http://")
    body, content_type = _multipart(
        {"speaker_id": "omar", "consent": "agreed"},
        {"audio": ("a.wav", audio_bytes(clips["omar"][0]))},
    )
    assert len(body) > 1024, "the fixture must be big enough to trigger the header"

    conn = http.client.HTTPConnection(host, timeout=60)
    conn.request(
        "POST", "/api/enroll", body=body,
        headers={"Content-Type": content_type, "Content-Length": str(len(body)),
                 "Expect": "100-continue"},
    )
    response = conn.getresponse()
    payload = json.loads(response.read())
    conn.close()
    assert response.status == 200
    assert payload["speaker_id"] == "omar"


def test_server_speaks_http_1_1(server):
    """HTTP/1.1 is required for the Expect handling above and for keep-alive."""
    import http.client

    base, _ = server
    conn = http.client.HTTPConnection(base.removeprefix("http://"), timeout=30)
    conn.request("GET", "/api/status")
    response = conn.getresponse()
    response.read()
    conn.close()
    assert response.version == 11


def test_calibrate_response_is_strict_json_and_flags_unusability(server):
    """One take per speaker means nothing was measured; say so, in valid JSON."""
    base, clips = server
    for name in ("omar", "hana"):
        post(base, "/api/enroll", {"speaker_id": name, "consent": "agreed"},
             {"audio": ("a.wav", audio_bytes(clips[name][0]))})

    _, body, _ = post(base, "/api/calibrate")

    def reject(constant):
        raise AssertionError(f"non-JSON constant in response: {constant}")

    payload = json.loads(body, parse_constant=reject)
    assert payload["usable"] is False
    assert payload["eer"] is None
    assert payload["warnings"]
    assert get_json(base, "/api/status")["threshold"] is None
