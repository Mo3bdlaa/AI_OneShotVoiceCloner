# voxprint

Fingerprint a voice from a short recording, recognise it later, and imitate it.

Three capabilities, deliberately kept apart because they differ enormously in how
well they actually work:

| | what it does | how well it works |
|---|---|---|
| **fingerprint** | turn a recording into a fixed-length voice print | works, with real caveats |
| **recognise** | match a new recording against enrolled speakers, or say "unknown" | works, with real caveats |
| **imitate** | speak new text in that voice | needs a pretrained model — no signal-processing shortcut exists |

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

# 4. imitate
voxprint convert someone.wav --id omar -o out.wav              # works offline
voxprint speak "مرحبا" --id omar --language ar -o hello.wav \
    --backend xtts                                             # needs the neural extras

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

Every table here is produced by `python tools/benchmark.py`, on 12 enrolled
speakers and 12 strangers drawn from one mutually distinct pool of procedural
voices, with 4-second takes and held-out queries.

| encoder | EER | closed-set | open-set rejection | same-speaker | impostor |
|---|---|---|---|---|---|
| `dsp` | 0.000 | 91.7 % | 100 % | +0.94 | +0.01 |
| `ecapa` | 0.000 | 100 % | 91.7 % | +0.95 | +0.48 |

**These are an upper bound**, and the fixture matters as much as the result: the
voices are clean and perfectly matched in recording conditions. Three limits
matter, all measured rather than guessed.

### Noise degrades scores long before it degrades ranking

Clean enrolment, query degraded with pink noise:

| query | `dsp` rank-1 | `dsp` accepted | `ecapa` rank-1 | `ecapa` accepted |
|---|---|---|---|---|
| clean | 100 % | 91.7 % | 100 % | 100 % |
| 40 dB SNR | 100 % | 2.8 % | 100 % | 41.7 % |
| 30 dB SNR | 86.1 % | 0 % | 77.8 % | 0 % |
| 20 dB SNR | 52.8 % | 0 % | 86.1 % | 0 % |
| 10 dB SNR | 50.0 % | 0 % | 91.7 % | 0 % |

Two separate things are visible here, and conflating them sends you to the wrong
fix.

*The encoder.* ECAPA keeps ranking the right speaker first down to 10 dB SNR
(91.7 %) where the DSP encoder is at chance-ish 50 %. That gap is the difference
between a network trained discriminatively on thousands of speakers and a
hand-built statistic, and it is why `--encoder ecapa` is the answer when accuracy
matters. (ECAPA is also *understated* here: it was trained on real human speech,
and these synthetic vowel-only voices are out of its domain.)

*The threshold.* Acceptance collapses to zero for **both** encoders, because both
were given a threshold derived from clean-against-clean comparisons and any
condition mismatch shifts every score down. No encoder fixes that. Calibration
does:

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

### On real speech

The numbers above come from synthetic voices, which is a statement about the
pipeline, not about accuracy on people. Point the harness at a real corpus laid
out one folder per speaker:

```bash
voxprint eval /path/to/corpus --enroll-files 3
```

It holds whole speakers out as strangers (not spare clips of enrolled speakers),
reports a trial EER from held-out queries alongside the optimistic
enrolment-based one, and separates ranking accuracy from acceptance. VoxCeleb,
LibriSpeech, Common Voice grouped by `client_id`, or your own recordings all work.

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
  synth/            imitation backends (dspvc, xtts, yourtts)
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
by the standard deviation of a reference population changes that to 0.82–0.89
against below 0.57. `voxprint/normalization.py` explains why, and
`tools/build_reference.py` rebuilds the shipped statistics — point it at a real
corpus if you have one.

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
| `dspvc` | speech → speech | nothing | timbre transfer, not identity cloning |
| `xtts` | text → speech, speech → speech | neural extras, ~2 GB | 17 languages including Arabic; ~1.4× real time on CPU; **non-commercial licence** |
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

242 tests, about a minute, no network access and no downloads — all test audio is
generated (the three that pull a 2 GB checkpoint are opt-in and skip by default). Several of them pin the claims in this README: measured accuracy, the
watermark's detection margin *and* its documented failure under telephone
bandwidth, the fact that robust calibration lowers the threshold without lowering
clean acceptance. A regression fails the suite rather than quietly outdating the
docs.

The optional backends have their own file, skipped when their dependencies are
absent:

```bash
pip install -r requirements-neural.txt
COQUI_TOS_AGREED=1 VOXPRINT_TEST_XTTS=1 pytest tests/test_neural_backends.py
```

`python tools/benchmark.py --encoders dsp ecapa` reproduces every table above.

## Licence

MIT (see `LICENSE`). Pretrained models downloaded by the optional backends carry
their own licences — XTTS-v2 in particular is non-commercial.
