# voxprint

Fingerprint a voice from a short recording, recognise it later, and imitate it.

Three capabilities, deliberately kept apart because they differ enormously in how
well they actually work:

| | what it does | how well it works |
|---|---|---|
| **fingerprint** | turn a recording into a fixed-length voice print | works, with real caveats |
| **recognise** | match a new recording against enrolled speakers, or say "unknown" | works, with real caveats |
| **re-voice** | take a recording and change whose voice it is, keeping the words | works well — needs a pretrained model |
| **speak** | generate new speech from text in that voice | works — needs a pretrained model |

Everything except imitation runs on NumPy and SciPy alone: no model download, no
GPU, no licence to read. Imitation is an optional backend, because doing it
properly means loading weights that somebody trained on thousands of hours of
speech.

Every number below was measured by code in this repository. Run
`voxprint selftest` and you will get numbers of the same kind on your machine.

---

## Install

```bash
git clone https://github.com/Mo3bdlaa/AI_OneShotVoiceCloner
cd AI_OneShotVoiceCloner
pip install -e .
```

That is the whole install for fingerprinting and recognition. For voice cloning
from text:

```bash
pip install -r requirements-neural.txt   # PyTorch + SpeechBrain + Coqui TTS
```

## Try it without any audio files

```bash
python examples/demo.py --out demo_output
```

The demo builds a cast of procedurally generated voices, enrols three of them,
calibrates a threshold, identifies held-out takes, rejects a speaker it has never
heard, converts one voice toward another, and measures how far the conversion
actually got. No recordings of real people are involved.

## Use it on real audio

```bash
# 1. enrol -- several takes are much better than one
voxprint enroll omar_1.wav omar_2.wav omar_3.wav \
    --id omar --consent "agreed to voice enrolment on 2026-01-04"

# 2. calibrate -- turns measured scores into a decision threshold
voxprint calibrate

# 3. recognise
voxprint identify unknown.wav          # exit 0 = match, 2 = unknown
voxprint verify claim.wav --id omar    # exit 0 = accept, 2 = reject

# 4a. re-voice an existing recording: keeps the words, timing and performance
voxprint revoice recording.wav --id omar -o out.wav            # needs the neural extras
voxprint revoice song.mp3 --id omar -o out.wav                 # splits the vocal off first

# 4b. generate new speech from text
voxprint speak "مرحبا" --id omar --language ar -o hello.wav \
    --backend xtts

# or drive all of it from a browser, recording straight from the microphone
voxprint serve
```

From Python:

```python
from voxprint import VoiceLab, ConsentRecord

lab = VoiceLab("voices")
lab.enroll("omar", ["omar_1.wav", "omar_2.wav"],
           consent=ConsentRecord(granted_by="Omar", statement="agreed 2026-01-04"))
lab.calibrate()

result = lab.identify_file("unknown.wav")
print(result.decision, result.best.speaker_id, result.best.score)
```

---

## How well does it actually work?

### On real speech

Measured on LibriSpeech `dev-clean`: 30 enrolled speakers, 3 enrolment files
each, 150 held-out queries, and 10 speakers held out entirely as strangers.

| encoder | EER (1-to-1) | EER (open-set) | correct | strangers rejected |
|---|---|---|---|---|
| `dsp` | 0.131 | 0.333 | 58 % | 88 % |
| **`ecapa`** | **0.011** | **0.033** | **100 %** | **100 %** |

Reproduce with `voxprint eval /path/to/LibriSpeech/dev-clean --encoder ecapa`.

That 1.1 % equal error rate is the published VoxCeleb-class figure, measured here
rather than cited, and it is the honest answer to "is this usable?". **Yes, with
`--encoder ecapa`.** The built-in `dsp` encoder gets a bit over half the
identifications right on real speech — useful as a baseline and for testing the
pipeline, not for anything that matters.

### On synthetic voices

`python tools/benchmark.py` runs the same protocol on 12 procedural speakers and
12 strangers from one mutually distinct pool. It is an upper bound and a
regression check, not a claim about people:

| encoder | EER | closed-set | open-set rejection | same-speaker | impostor |
|---|---|---|---|---|---|
| `dsp` | 0.000 | 91.7 % | 100 % | +0.96 | +0.40 |

Three limits matter, all measured rather than guessed.

### Noise degrades scores long before it degrades ranking

Clean enrolment, query degraded with pink noise:

| query | `dsp` rank-1 | `dsp` accepted |
|---|---|---|
| clean | 100 % | 91.7 % |
| 40 dB SNR | 100 % | 5.6 % |
| 30 dB SNR | 91.7 % | 0 % |
| 20 dB SNR | 91.7 % | 0 % |
| 10 dB SNR | 75.0 % | 0 % |

Acceptance collapses long before ranking does, and the same holds for `ecapa`,
which still ranks the right speaker first 92 % of the time at 10 dB SNR. The
cause is not the encoder: a threshold derived from clean-against-clean
comparisons is invalidated by any condition mismatch, which shifts every score
down. Calibration fixes it:

```bash
voxprint calibrate --robust
```

which also scores degraded copies of the enrolment audio — pink and white noise,
reverberation, telephone bandwidth, a microphone tilt. Measured on five
speakers, acceptance of queries at 30 dB SNR goes from 50 % to 83 %, with clean
acceptance unchanged; the equal error rate rises from 0.058 to 0.102, which is
the honest price of a threshold that survives a change of room.

### Short queries are unreliable

Under one second there is not enough speech to estimate the statistics: 1 s
fails outright, 2 s reaches 57 %, 3 s and above 95 % or better. Aim for three
seconds of query and six or more for enrolment.

### Cross-microphone matching is the weak point by design

Half the `dsp` feature blocks describe the absolute spectrum, which the
microphone and the room colour. Enrolling on a phone and querying on a laptop is
the case it handles worst — use `--encoder ecapa` and `calibrate --robust`.

### Identification and verification need different thresholds

Verification compares one score against one claim. Identification takes the
**maximum** over every enrolled speaker, and the maximum of N draws sits far
above a single draw: measured on 30 real speakers, an unenrolled voice's best
match averages +0.354 against a pairwise impostor mean of −0.020. Calibrating
identification with a pairwise threshold admitted 49 of 50 strangers.

`voxprint calibrate` therefore measures both — pairwise trials for verification,
and a leave-one-speaker-out best-match distribution for identification — and
`identify` uses the second. On the same data that lifted stranger rejection from
2 % to 70 %, at a cost of 9 points of accuracy. Recalibrate after the gallery
grows: the maximum is over more candidates.

### Re-voicing works, and it beats the recogniser

Give it a recording and a target voice, and it returns the same words in that
voice. Measured between two LibriSpeech speakers, scoring the output against the
target's **held-out** ECAPA voice print (the source scored −0.04 to the target
and +0.92 to itself beforehand):

| backend | → target | → source | time |
|---|---|---|---|
| **`knnvc`** (default) | **+0.645** | **+0.031** | 8 s |
| `openvoice` | +0.401 | +0.265 | 6 s |
| `freevc` | +0.239 | +0.161 | 33 s |
| `dspvc` (no download) | — closes about half the gap, never crosses it — | | instant |

kNN-VC matches each frame of the source against the target's own recordings, so
unlike encoder-based converters it keeps improving with more reference audio:

| reference | 5 s | 10 s | 20 s | 40 s |
|---|---|---|---|---|
| → target | +0.531 | +0.647 | +0.732 | +0.776 |

Twenty to forty seconds is the useful range; `voxprint enroll` keeps up to 60 s
for exactly this. The source identity is gone throughout (+0.02 to +0.04).

**The full loop, end to end.** Taking a recording of speaker A over a backing
track, re-voicing it toward speaker B with 47 s of B's audio, and then asking the
system itself who is speaking:

```
  the output scores +0.629 against 1988's voice print   (threshold 0.554)
  MATCH: 1988   1. 1988 +0.617   2. 1272 +0.003
```

The recogniser identifies the re-voiced audio as the target and puts the original
speaker at +0.003. That is the demonstration, and it is also the reason
`docs/ETHICS.md` says not to authorise anything with a voice.

### Songs

`voxprint revoice song.mp3` separates the vocal with Demucs, converts it, and
mixes it back over the untouched instrumental. On the test mix the separated
vocal correlated +0.968 with the true vocal and +0.064 with the backing, and the
re-voiced result scored +0.668 mixed, +0.700 as a bare vocal.

**But singing is not speech, and every converter here was trained on speech.**
Sustained vowels, vibrato and a two-octave range are out of domain; separation
adds artefacts that conversion then amplifies. Expect a usable result on speech
and a rough one on singing. Systems built for singing — RVC, so-vits-svc — sound
far better and are *not* zero-shot: they want ~10 minutes of the target voice and
a training run. This pipeline is the right shape for those too; swap the
converter and keep the separation and remix.

On CPU, budget roughly 4× real time for separation and 2× for conversion.

### Measure on your own data

```bash
voxprint eval /path/to/corpus --enroll-files 3 --encoder ecapa
```

One folder per speaker. VoxCeleb, LibriSpeech, Common Voice grouped by
`client_id`, or your own recordings.

`docs/FEASIBILITY.md` goes through what each part of the system can and cannot
deliver, and why.

---

## What is inside

```
voxprint/
  audio.py          I/O, resampling, loudness, silence removal, VAD
  features.py       mel filterbank, MFCC, LPC cepstrum, pitch tracking
  spectral.py       STFT, phase vocoder, pitch shifting
  encoders/         voice-print extractors (dsp, ecapa, resemblyzer)
  normalization.py  the step that makes cosine scoring work at all
  gallery.py        enrolled speakers, consent records, persistence
  scoring.py        identification, verification, threshold calibration
  synth/            imitation backends (knnvc, openvoice, freevc, dspvc, xtts)
  song.py           separate a vocal, re-voice it, mix it back
  augment.py        noise, reverb, channel -- for realistic calibration
  watermark.py      provenance marking of generated audio
  pipeline.py       VoiceLab -- the API the CLI is built on
  evaluate.py       self-evaluation on procedural voices
  corpus.py         evaluation on a real corpus
  server.py         HTTP API and browser UI
```

Two design decisions are worth knowing about before reading the code.

**Standardisation is not optional.** Raw statistics-based voice prints all sit
within a hundredth of 1.0 of each other: measured same-speaker cosine 0.99 against
different-speaker 0.98, which no threshold can separate. Dividing each dimension
by the standard deviation of a reference population fixes it, and *which*
population matters: on real speech, a reference fitted on procedural voices alone
rejects 60 % of strangers, one fitted on LibriSpeech 82 %, and one pooled from
both 88 %. The shipped reference is the pooled one. `tools/build_reference.py`
rebuilds it — point it at a corpus that matches your recording conditions.

**Thresholds are measured, not chosen.** `voxprint calibrate` scores every
enrolment take against the leave-one-out centroid of its own speaker and against
every other speaker, then finds the equal-error point. A hard-coded threshold
would be a guess about your microphone, your room, and your speakers. Add
`--robust` and the trials include degraded copies of the enrolment audio, which
is what makes the threshold survive a condition change.

## Encoders and backends

```bash
voxprint backends    # shows what is installed and what each thing costs
```

| encoder | dims | needs | notes |
|---|---|---|---|
| `dsp` | 373 | nothing | the default; works everywhere, weak under noise |
| `ecapa` | 192 | PyTorch + SpeechBrain | ~1 % EER on VoxCeleb, ~80 MB download |
| `resemblyzer` | 256 | PyTorch + Resemblyzer | lighter GE2E d-vectors |

| backend | can do | needs | notes |
|---|---|---|---|
| `knnvc` | speech → speech | neural extras | **best identity transfer**; improves with more reference audio |
| `openvoice` | speech → speech | neural extras | MIT-licensed weights, weaker transfer |
| `freevc` | speech → speech | neural extras | 24 kHz output, weakest transfer of the three |
| `dspvc` | speech → speech | nothing | timbre transfer only, but instant and offline |
| `xtts` | text → speech | neural extras, ~2 GB | 17 languages including Arabic; ~1.4× real time on CPU; **non-commercial licence** |
| `yourtts` | text → speech | neural extras | lighter, lower fidelity, en/fr/pt |

Both `ecapa` and `xtts` have been run end to end, not just imported — including
Arabic synthesis. The measurements, and the two dependency pins needed to make
Coqui TTS import at all, are in `docs/FEASIBILITY.md`. XTTS will not run until
you set `COQUI_TOS_AGREED=1`: the checkpoint is non-commercial, and this code does
not accept that licence on your behalf.

Switching encoder changes nothing else — the gallery refuses to mix embeddings
from different encoders rather than silently comparing incomparable vectors.

## Browser UI

```bash
voxprint serve        # http://127.0.0.1:8000
```

Record from the microphone, enrol, calibrate, identify, convert, check a
watermark. Binds to localhost by default — the gallery is biometric data and the
server has no authentication, so exposing it is a decision you have to make
explicitly with `--host`.

## Consent and provenance

Enrolment requires a consent record naming who agreed and to what. A voice print
is biometric data, and the stored reference audio can drive the cloning backends.
`--no-consent-check` exists for synthetic voices and test fixtures.

Generated audio is watermarked by default: a spread-spectrum mark 34 dB below the
signal, detectable at z ≥ 27 where unmarked audio peaks at 5.8. It survives
cropping, added noise and band-limiting to 12 kHz, and it does **not** survive
telephone bandwidth. It is a provenance marker for a cooperative verifier, not a
defence against an adversary. See `docs/ETHICS.md`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

257 tests, about 85 seconds, no network access and no downloads — all test audio is
generated (the three that pull a 2 GB checkpoint are opt-in and skip by default). Several of them pin the claims in this README: measured accuracy, the
watermark's detection margin *and* its documented failure under telephone
bandwidth, the fact that robust calibration lowers the threshold without lowering
clean acceptance. A regression fails the suite rather than quietly outdating the
docs.

The optional backends have their own file, skipped when their dependencies are
absent:

```bash
pip install -r requirements-neural.txt     # includes demucs, for songs
COQUI_TOS_AGREED=1 VOXPRINT_TEST_XTTS=1 pytest tests/test_neural_backends.py
```

`python tools/benchmark.py --encoders dsp ecapa` reproduces every table above.

## Licence

MIT (see `LICENSE`). Two things it does not cover:

- **Pretrained models** downloaded by the optional backends carry their own
  licences — XTTS-v2 in particular is non-commercial.
- **The shipped reference statistics** (`voxprint/data/reference_dsp_v1.npz`) are
  derived in part from LibriSpeech `dev-other`, which is CC BY 4.0
  (V. Panayotov, G. Chen, D. Povey and S. Khudanpur, *LibriSpeech: an ASR corpus
  based on public domain audio books*, ICASSP 2015). The file holds two
  373-element vectors of per-dimension means and standard deviations — no audio
  and nothing speaker-identifiable. Rebuild it from your own data with
  `tools/build_reference.py`.
