"""HTTP server and browser UI.

A small wrapper over :class:`voxprint.pipeline.VoiceLab` so the system can be
used from a browser -- record straight from the microphone, enrol, identify,
convert -- without touching the command line.

Deliberate constraints
----------------------
* **Local by default.** It binds to ``127.0.0.1``. A voice-print gallery is
  biometric data; exposing it on a network is a decision the operator has to make
  explicitly with ``--host``, and the server says so when they do.
* **The same consent rule as everywhere else.** Enrolment through the API
  requires a consent statement, exactly as the CLI does.
* **No new logic.** Every endpoint maps to a ``VoiceLab`` method. Anything the
  server can do, the CLI and the Python API can do identically.

Implemented on ``http.server`` from the standard library rather than a web
framework, so the core install stays at three dependencies.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from email.parser import BytesParser
from email.policy import default as email_default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .audio import load_audio, save_audio
from .gallery import ConsentRecord, GalleryError
from .pipeline import VoiceLab
from .watermark import detect_watermark

MAX_UPLOAD_BYTES = 512 * 1024 * 1024   # a training set is a folder, not a clip


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def json_default(obj):
    """Convert NumPy scalars to JSON-native types.

    Passing ``default=str`` instead is a trap: a ``numpy.bool_`` then serialises
    as the string ``"True"``, which is truthy in every client language and so
    fails silently rather than loudly.
    """
    import numpy as np

    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


@dataclass
class _Job:
    """A long-running task the browser polls rather than waits for.

    Training takes hours. An HTTP request cannot hold that open, and a progress
    bar that lies is worse than none, so the work runs on a thread and the page
    asks how it is going.
    """

    id: str
    kind: str
    speaker: str
    state: str = "running"          # running | done | failed
    started: float = field(default_factory=time.time)
    finished: float | None = None
    detail: str = ""
    result: dict | None = None
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "speaker": self.speaker,
            "state": self.state,
            "seconds": round((self.finished or time.time()) - self.started, 1),
            "detail": self.detail,
            "result": self.result,
            "error": self.error,
        }


class _Api:
    """Thread-safe façade over one VoiceLab."""

    def __init__(self, lab: VoiceLab):
        self.lab = lab
        self._lock = threading.Lock()
        self._jobs: dict[str, _Job] = {}

    # -- endpoints --------------------------------------------------------- #

    def status(self) -> dict:
        with self._lock:
            return self.lab.summary()

    def speakers(self) -> list[dict]:
        with self._lock:
            return [
                p.quality_report() | {"display_name": p.display_name, "has_consent": p.consent is not None}
                for p in self.lab.speakers()
            ]

    def enroll(self, speaker_id: str, audio: bytes, consent: str, granted_by: str) -> dict:
        if not consent.strip():
            raise ValueError(
                "a consent statement is required: record who agreed to have this voice enrolled"
            )
        with self._lock:
            path = self._stash(audio, f"enroll_{speaker_id}")
            try:
                report = self.lab.enroll(
                    speaker_id,
                    [path],
                    consent=ConsentRecord(granted_by=granted_by or speaker_id, statement=consent),
                )
                return report.as_dict()
            finally:
                os.remove(path)

    def identify(self, audio: bytes) -> dict:
        with self._lock:
            path = self._stash(audio, "identify")
            try:
                return self.lab.identify_file(path).as_dict()
            finally:
                os.remove(path)

    def convert(self, audio: bytes, speaker_id: str, backend: str = "dspvc") -> tuple[bytes, dict]:
        with self._lock:
            src = self._stash(audio, "convert")
            try:
                result = self.lab.convert_file(src, speaker_id=speaker_id, backend=backend)
                out = self._stash(b"", "converted")
                save_audio(out, result.wav, result.sample_rate)
                with open(out, "rb") as fh:
                    data = fh.read()
                os.remove(out)
                info = dict(result.info)
                # Name the backend that produced this: the caller chose one, and
                # a silent fallback would otherwise be indistinguishable.
                info["backend"] = result.backend
                info["sample_rate"] = result.sample_rate
                if speaker_id in self.lab.gallery:
                    info["similarity_to_target"] = round(
                        self.lab.similarity_to(result.wav, result.sample_rate, speaker_id), 4
                    )
                    info["threshold"] = self.lab.gallery.threshold
                return data, info
            finally:
                os.remove(src)

    def calibrate(self, robust: bool = False) -> dict:
        with self._lock:
            return self.lab.calibrate(robust=robust).as_dict()

    def remove(self, speaker_id: str) -> dict:
        with self._lock:
            self.lab.remove(speaker_id)
            return {"removed": speaker_id}

    # -- long-running jobs -------------------------------------------------- #

    def jobs(self) -> list[dict]:
        with self._lock:
            return [j.as_dict() for j in sorted(self._jobs.values(), key=lambda j: -j.started)]

    def backends(self) -> dict:
        """What each conversion backend can do right now, and what it needs."""
        from .svc import available as svc_available
        from .synth import available_synths, get_synth

        out = []
        for name in available_synths():
            try:
                backend = get_synth(name)
                ready, reason = backend.available()
                info = {"name": name, "ready": ready, "status": reason,
                        "capabilities": sorted(backend.capabilities), "notes": backend.notes}
            except Exception as exc:
                info = {"name": name, "ready": False, "status": str(exc),
                        "capabilities": [], "notes": ""}
            out.append(info)
        svc_ready, svc_reason = svc_available()
        return {"backends": out, "training": {"ready": svc_ready, "status": svc_reason}}

    def train(self, speaker_id: str, clips: list[bytes], epochs: int | None) -> dict:
        """Start a so-vits-svc training run for a speaker, in the background."""
        from .svc import MIN_TRAINING_MINUTES, train_for_speaker

        with self._lock:
            print_ = self.lab.gallery.get(speaker_id)       # raises if unknown
            if self.lab.gallery.require_consent and print_.consent is None:
                raise ValueError(
                    f"{speaker_id} has no consent record. A trained model can generate unlimited "
                    "audio in this voice; record consent before training one."
                )
            if any(j.state == "running" for j in self._jobs.values()):
                raise ValueError("a training run is already in progress")

        directory = tempfile.mkdtemp(prefix=f"voxprint_train_{_safe(speaker_id)}_")
        for index, data in enumerate(clips):
            with open(os.path.join(directory, f"{index:04d}.wav"), "wb") as fh:
                fh.write(data)

        job = _Job(id=uuid.uuid4().hex[:12], kind="train-svc", speaker=speaker_id,
                   detail=f"{len(clips)} clip(s) staged; preprocessing")

        def run() -> None:
            try:
                report = train_for_speaker(self.lab, speaker_id, directory, epochs=epochs)
                job.result = report.as_dict()
                job.state = "done"
                job.detail = (
                    f"trained on {report.minutes_of_audio:.1f} min"
                    + (f"; below the {MIN_TRAINING_MINUTES:.0f} min this wants"
                       if report.minutes_of_audio < MIN_TRAINING_MINUTES else "")
                )
            except Exception as exc:
                job.state = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
            finally:
                job.finished = time.time()
                shutil.rmtree(directory, ignore_errors=True)

        with self._lock:
            self._jobs[job.id] = job
        threading.Thread(target=run, daemon=True, name=f"train-{job.id}").start()
        return job.as_dict()

    def watermark(self, audio: bytes) -> dict:
        path = self._stash(audio, "watermark")
        try:
            wav, _ = load_audio(path)
            return detect_watermark(wav).as_dict()
        finally:
            os.remove(path)

    # -- helpers ----------------------------------------------------------- #

    def _stash(self, data: bytes, prefix: str) -> str:
        """Write an upload to a temp file: the audio decoders want a path."""
        import tempfile

        fd, path = tempfile.mkstemp(prefix=f"voxprint_{prefix}_", suffix=".wav")
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        return path


class _Handler(BaseHTTPRequestHandler):
    api: _Api = None  # set by serve()
    server_version = "voxprint"

    #: ``BaseHTTPRequestHandler`` defaults to HTTP/1.0, under which the server
    #: rejects the ``Expect: 100-continue`` header that clients send ahead of a
    #: body over about 1 KB. Every audio upload is larger than that, so with the
    #: default every POST carrying a file failed to connect at all -- while POSTs
    #: without a body, and all GETs, worked. Declaring HTTP/1.1 fixes it and
    #: enables keep-alive; it also makes an accurate ``Content-Length`` on every
    #: response mandatory, which the helpers below all set.
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter than the default
        if self.path != "/api/status":
            super().log_message(fmt, *args)

    def handle_expect_100(self) -> bool:
        """Acknowledge ``Expect: 100-continue`` so large uploads proceed."""
        self.send_response_only(100)
        self.end_headers()
        return True

    # -- responses --------------------------------------------------------- #

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _audio(self, data: bytes, info: dict) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Voxprint-Info", json.dumps(info, default=json_default))
        self.end_headers()
        self.wfile.write(data)

    def _html(self, body: str) -> None:
        raw = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    # -- routing ----------------------------------------------------------- #

    def do_GET(self):
        route = urlparse(self.path).path
        try:
            if route in ("/", "/index.html"):
                return self._html(PAGE)
            if route == "/api/status":
                return self._json(self.api.status())
            if route == "/api/speakers":
                return self._json(self.api.speakers())
            if route == "/api/backends":
                return self._json(self.api.backends())
            if route == "/api/jobs":
                return self._json(self.api.jobs())
            return self._json({"error": "not found"}, 404)
        except Exception as exc:
            return self._json({"error": str(exc)}, 500)

    def do_POST(self):
        route = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_UPLOAD_BYTES:
                return self._json({"error": f"upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"}, 413)
            raw = self.rfile.read(length) if length else b""
            fields, files = _parse_multipart(self.headers.get("Content-Type", ""), raw)

            if route == "/api/enroll":
                return self._json(
                    self.api.enroll(
                        _require(fields, "speaker_id"),
                        _require(files, "audio"),
                        fields.get("consent", ""),
                        fields.get("granted_by", ""),
                    )
                )
            if route == "/api/identify":
                return self._json(self.api.identify(_require(files, "audio")))
            if route == "/api/convert":
                data, info = self.api.convert(
                    _require(files, "audio"),
                    _require(fields, "speaker_id"),
                    fields.get("backend") or "dspvc",
                )
                return self._audio(data, info)
            if route == "/api/train":
                clips = [v for k, v in files.items() if k.startswith("audio")]
                if not clips:
                    raise ValueError("no training audio supplied")
                epochs = int(fields["epochs"]) if fields.get("epochs") else None
                return self._json(self.api.train(_require(fields, "speaker_id"), clips, epochs))
            if route == "/api/watermark":
                return self._json(self.api.watermark(_require(files, "audio")))
            if route == "/api/calibrate":
                query = parse_qs(urlparse(self.path).query)
                return self._json(self.api.calibrate(robust=query.get("robust", ["0"])[0] == "1"))
            if route == "/api/remove":
                return self._json(self.api.remove(_require(fields, "speaker_id")))
            return self._json({"error": "not found"}, 404)
        except (ValueError, KeyError, GalleryError) as exc:
            return self._json({"error": str(exc)}, 400)
        except Exception as exc:
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)


def _require(mapping, key):
    if key not in mapping or not mapping[key]:
        raise ValueError(f"missing required field {key!r}")
    return mapping[key]


def _parse_multipart(content_type: str, body: bytes) -> tuple[dict[str, str], dict[str, bytes]]:
    """Split a multipart/form-data body into text fields and file parts."""
    if not content_type.startswith("multipart/form-data"):
        return {}, {}
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
    message = BytesParser(policy=email_default).parsebytes(header + body)

    fields: dict[str, str] = {}
    files: dict[str, bytes] = {}
    for part in message.iter_parts():
        disposition = part.get("Content-Disposition", "")
        match = re.search(r'name="([^"]+)"', disposition)
        if not match:
            continue
        name = match.group(1)
        payload = part.get_payload(decode=True) or b""
        if "filename=" in disposition:
            # Several files can share one field name -- training takes a whole
            # folder of clips. Later ones are suffixed rather than overwriting,
            # and the train route collects every key starting with its name.
            key = name if name not in files else f"{name}{len(files)}"
            files[key] = payload
        else:
            fields[name] = payload.decode("utf-8", errors="replace").strip()
    return fields, files


def serve(
    root: str = "voices",
    encoder: str = "dsp",
    host: str = "127.0.0.1",
    port: int = 8000,
    require_consent: bool = True,
) -> None:
    """Start the server. Blocks until interrupted."""
    lab = VoiceLab(root, encoder=encoder, require_consent=require_consent)
    _Handler.api = _Api(lab)
    httpd = ThreadingHTTPServer((host, port), _Handler)

    print(f"voxprint server on http://{host}:{port}  (gallery: {root}, encoder: {lab.encoder.name})")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(
            "  ! bound to a non-local address. The gallery holds biometric data and this server\n"
            "    has no authentication -- put it behind one, or bind to 127.0.0.1."
        )
    print("  press Ctrl-C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>voxprint</title>
<style>
  :root {
    --bg: #fbfbfa; --panel: #fff; --ink: #1a1a19; --muted: #6b6b68;
    --line: #e4e4e1; --accent: #2f6f4e; --warn: #9a5b12; --bad: #a33a2a;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #17171a; --panel: #1f1f23; --ink: #ececea; --muted: #9a9a96;
      --line: #313137; --accent: #6fbf94; --warn: #d5a15c; --bad: #e08272;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 0 16px 64px; background: var(--bg); color: var(--ink);
    font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  .wrap { max-width: 760px; margin: 0 auto; }
  header { padding: 32px 0 8px; }
  h1 { font-size: 22px; margin: 0 0 4px; letter-spacing: -0.01em; }
  .sub { color: var(--muted); font-size: 13px; margin: 0; }
  section {
    background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
    padding: 18px; margin: 16px 0;
  }
  h2 { font-size: 14px; margin: 0 0 12px; text-transform: uppercase;
       letter-spacing: 0.06em; color: var(--muted); font-weight: 600; }
  label { display: block; font-size: 13px; color: var(--muted); margin: 10px 0 4px; }
  input, select, textarea {
    width: 100%; padding: 8px 10px; font: inherit; color: var(--ink);
    background: var(--bg); border: 1px solid var(--line); border-radius: 6px;
  }
  textarea { min-height: 54px; resize: vertical; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
  button {
    font: inherit; padding: 8px 14px; border-radius: 6px; cursor: pointer;
    border: 1px solid var(--line); background: var(--bg); color: var(--ink);
  }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  button.rec { border-color: var(--bad); color: var(--bad); }
  button.rec[data-on="1"] { background: var(--bad); color: #fff; }
  pre {
    background: var(--bg); border: 1px solid var(--line); border-radius: 6px;
    padding: 10px; overflow-x: auto; font-size: 12.5px; margin: 12px 0 0;
    white-space: pre-wrap; word-break: break-word;
  }
  table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); }
  th { color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; }
  .note { font-size: 12.5px; color: var(--muted); margin-top: 10px; }
  .warn { color: var(--warn); }
  audio { width: 100%; margin-top: 12px; }
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>voxprint</h1>
  <p class="sub" id="status">loading…</p>
</header>

<section>
  <h2>1 · record or choose audio</h2>
  <div class="row">
    <button class="rec" id="rec" data-on="0">● record</button>
    <input type="file" id="file" accept="audio/*" style="flex:1; min-width:200px">
  </div>
  <audio id="preview" controls hidden></audio>
  <p class="note">Recording needs a secure context: works on <code>localhost</code>, otherwise HTTPS.</p>
</section>

<section>
  <h2>2 · enrol</h2>
  <label for="sid">speaker id</label>
  <input id="sid" placeholder="omar">
  <label for="consent">consent statement (required)</label>
  <textarea id="consent" placeholder="Omar agreed to have his voice enrolled on 2026-01-04, for device unlock testing."></textarea>
  <div class="row"><button class="primary" id="doEnroll">enrol this clip</button></div>
  <pre id="enrollOut" hidden></pre>
</section>

<section>
  <h2>3 · enrolled speakers</h2>
  <table id="table"><thead><tr><th>id</th><th>takes</th><th>seconds</th><th>cohesion</th><th>consent</th><th></th></tr></thead><tbody></tbody></table>
  <div class="row">
    <button id="doCal">calibrate</button>
    <button id="doCalRobust">calibrate (robust)</button>
  </div>
  <p class="note">A clean calibration only fits queries recorded like the enrolment was.
     <em>Robust</em> also scores degraded copies — noise, room, telephone band — so the
     threshold survives a change of microphone, at the cost of more false accepts.</p>
  <pre id="calOut" hidden></pre>
</section>

<section>
  <h2>4 · identify</h2>
  <div class="row"><button class="primary" id="doId">who is this?</button>
                   <button id="doWm">check watermark</button></div>
  <pre id="idOut" hidden></pre>
</section>

<section>
  <h2>5 · convert toward a voice</h2>
  <label for="target">target speaker</label>
  <select id="target"></select>
  <label for="backend">converter</label>
  <select id="backend"></select>
  <p class="note" id="backendNote"></p>
  <div class="row"><button id="doConv">convert</button></div>
  <audio id="convAudio" controls hidden></audio>
  <pre id="convOut" hidden></pre>
  <p class="note">The result reports how close the output actually got to the target's
     voice print, judged by the gallery's own encoder.</p>
</section>

<section>
  <h2>6 · train a singing model</h2>
  <p class="note" id="trainStatus">checking…</p>
  <label for="trainTarget">speaker</label>
  <select id="trainTarget"></select>
  <label for="trainFiles">recordings of that voice — ten minutes or more, clean and solo</label>
  <input type="file" id="trainFiles" accept="audio/*" multiple>
  <label for="epochs">epochs (blank uses the config default)</label>
  <input id="epochs" type="number" min="1" placeholder="e.g. 100">
  <div class="row"><button id="doTrain">start training</button></div>
  <pre id="trainOut" hidden></pre>
  <p class="note">
    Only needed for <strong>singing</strong>. For speech the zero-shot converters are
    better and need no training at all — measured, kNN-VC reached +0.70 against the
    target's voice print where a trained so-vits-svc model peaked at +0.55 after
    6.5 hours. Training runs in the background; this page polls it. On a CPU it takes
    hours and the result will not be worth using — that part wants a GPU.
  </p>
</section>
</div>

<script>
let clip = null, recorder = null, chunks = [];

const $ = (id) => document.getElementById(id);
const show = (el, text) => { el.hidden = false; el.textContent = text; };

async function api(path, opts) {
  const res = await fetch(path, opts);
  const type = res.headers.get("Content-Type") || "";
  if (type.startsWith("audio/")) {
    return { blob: await res.blob(), info: JSON.parse(res.headers.get("X-Voxprint-Info") || "{}") };
  }
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function form(extra) {
  if (!clip) throw new Error("record or choose an audio clip first");
  const fd = new FormData();
  fd.append("audio", clip, "clip.wav");
  for (const [k, v] of Object.entries(extra || {})) fd.append(k, v);
  return fd;
}

function setClip(blob) {
  clip = blob;
  const url = URL.createObjectURL(blob);
  $("preview").src = url;
  $("preview").hidden = false;
}

$("file").onchange = (e) => { if (e.target.files[0]) setClip(e.target.files[0]); };

$("rec").onclick = async () => {
  const btn = $("rec");
  if (btn.dataset.on === "1") { recorder.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder = new MediaRecorder(stream);
    chunks = [];
    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      setClip(new Blob(chunks, { type: recorder.mimeType }));
      btn.dataset.on = "0"; btn.textContent = "● record";
    };
    recorder.start();
    btn.dataset.on = "1"; btn.textContent = "■ stop";
  } catch (err) {
    alert("microphone unavailable: " + err.message);
  }
};

let backendInfo = null;

async function refresh() {
  const [status, speakers, backends] = await Promise.all([
    api("/api/status"), api("/api/speakers"), api("/api/backends"),
  ]);
  backendInfo = backends;
  // Both thresholds, because they differ and identify uses the second one --
  // showing only the verification threshold next to an identify button misleads.
  const fmt = (v) => (v === null || v === undefined ? "not calibrated" : v.toFixed(3));
  $("status").textContent =
    `${status.speakers} speaker(s) · encoder ${status.encoder.name} (${status.encoder.dim} dims)` +
    ` · verify ${fmt(status.threshold)} · identify ${fmt(status.identification_threshold)}`;

  const sel = $("backend");
  if (!sel.options.length) {
    for (const b of backends.backends.filter((b) => b.capabilities.includes("vc"))) {
      const opt = document.createElement("option");
      opt.value = b.name;
      opt.textContent = b.ready ? b.name : `${b.name} (unavailable)`;
      opt.disabled = !b.ready;
      sel.appendChild(opt);
    }
    if ([...sel.options].some((o) => o.value === "knnvc" && !o.disabled)) sel.value = "knnvc";
    sel.onchange = showBackendNote;
  }
  showBackendNote();

  const t = backends.training;
  $("trainStatus").innerHTML = t.ready
    ? "so-vits-svc is installed."
    : `<span class="warn">unavailable: ${t.status}</span>`;
  $("doTrain").disabled = !t.ready;

  const body = $("table").querySelector("tbody");
  body.innerHTML = "";
  $("target").innerHTML = "";
  $("trainTarget").innerHTML = "";
  for (const s of speakers) {
    const tr = document.createElement("tr");
    const consent = s.has_consent
      ? "recorded"
      : '<span class="warn">none</span>';
    tr.innerHTML = `<td>${s.speaker_id}</td><td>${s.utterances}</td>` +
      `<td>${s.total_seconds.toFixed(1)}</td><td>${s.cohesion ?? "—"}</td><td>${consent}</td>` +
      `<td><button data-rm="${s.speaker_id}">delete</button></td>`;
    body.appendChild(tr);
    for (const id of ["target", "trainTarget"]) {
      const opt = document.createElement("option");
      opt.value = opt.textContent = s.speaker_id;
      $(id).appendChild(opt);
    }
  }
  body.querySelectorAll("[data-rm]").forEach((b) => {
    b.onclick = async () => {
      const fd = new FormData(); fd.append("speaker_id", b.dataset.rm);
      await api("/api/remove", { method: "POST", body: fd });
      refresh();
    };
  });
}

function wire(id, fn) {
  $(id).onclick = async () => {
    const btn = $(id); btn.disabled = true;
    try { await fn(); } catch (err) { alert(err.message); } finally { btn.disabled = false; }
  };
}

wire("doEnroll", async () => {
  const out = await api("/api/enroll", {
    method: "POST",
    body: form({ speaker_id: $("sid").value.trim(), consent: $("consent").value.trim() }),
  });
  show($("enrollOut"), JSON.stringify(out, null, 2));
  refresh();
});

wire("doId", async () => show($("idOut"), JSON.stringify(await api("/api/identify", { method: "POST", body: form() }), null, 2)));
wire("doWm", async () => show($("idOut"), JSON.stringify(await api("/api/watermark", { method: "POST", body: form() }), null, 2)));
wire("doCal", async () => { show($("calOut"), JSON.stringify(await api("/api/calibrate", { method: "POST" }), null, 2)); refresh(); });
wire("doCalRobust", async () => { show($("calOut"), JSON.stringify(await api("/api/calibrate?robust=1", { method: "POST" }), null, 2)); refresh(); });

wire("doConv", async () => {
  const { blob, info } = await api("/api/convert", {
    method: "POST",
    body: form({ speaker_id: $("target").value, backend: $("backend").value }),
  });
  $("convAudio").src = URL.createObjectURL(blob);
  $("convAudio").hidden = false;
  show($("convOut"), JSON.stringify(info, null, 2));
});

function showBackendNote() {
  if (!backendInfo) return;
  const b = backendInfo.backends.find((x) => x.name === $("backend").value);
  $("backendNote").textContent = b ? (b.ready ? b.notes : b.status) : "";
}

wire("doTrain", async () => {
  const files = $("trainFiles").files;
  if (!files.length) throw new Error("choose the recordings to train on");
  const fd = new FormData();
  fd.append("speaker_id", $("trainTarget").value);
  if ($("epochs").value) fd.append("epochs", $("epochs").value);
  for (const f of files) fd.append("audio", f, f.name);

  const job = await api("/api/train", { method: "POST", body: fd });
  show($("trainOut"), JSON.stringify(job, null, 2));
  pollJobs();
});

let polling = null;
async function pollJobs() {
  if (polling) return;
  polling = setInterval(async () => {
    let jobs;
    try { jobs = await api("/api/jobs"); } catch { return; }
    const running = jobs.filter((j) => j.state === "running");
    if (jobs.length) show($("trainOut"), JSON.stringify(jobs.slice(0, 3), null, 2));
    if (!running.length) {
      clearInterval(polling);
      polling = null;
      refresh();
    }
  }, 5000);
}

refresh();
pollJobs();
</script>
</body>
</html>
"""
