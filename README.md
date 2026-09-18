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

Measured on 20 procedurally generated speakers, three 4-second enrolment takes
each, scored on held-out recordings (`voxprint selftest` runs a smaller version):

| metric | result |
|---|---|
| equal error rate | 0.002 |
| closed-set accuracy (clean) | 98.3 % |
| open-set rejection of unenrolled voices | 100 % |
| same-speaker score | +0.91 |
| impostor score | −0.05 |

**These are an upper bound.** The voices are clean, perfectly matched in
recording conditions, and differ from each other in exactly the parameters the
encoder measures. Real speech is none of those things. Three limits matter, and
all three were measured rather than guessed:

**Noise breaks it.** With clean enrolment and a noisy query:

| query condition | ranking correct | accepted at the threshold |
|---|---|---|
| clean | 100 % | 98.3 % |
| 40 dB SNR | 93.3 % | 5.0 % |
| 30 dB SNR | 75.0 % | 0 % |
| 20 dB SNR | 23.3 % | 0 % |

Note the shape of that failure: at 40 dB the system still *ranks* the right
speaker first 93 % of the time, but every score falls below a threshold
calibrated on clean audio. Mild noise shows up first as a score shift, and only
later as a ranking collapse. The practical mitigation is to enrol under the same
conditions you will query under. The real fix is the `ecapa` encoder, which was
trained with noise augmentation.

**Short queries are unreliable.** Under one second there is not enough speech to
estimate the statistics: 1 s fails outright, 2 s gets 57 %, 3 s and above reach
95 % or better. Aim for at least three seconds, and six or more for enrolment.

**Cross-microphone matching is the weak point by design.** Half the feature
blocks describe the absolute spectrum, which the microphone and room colour.
Enrolling on a phone and querying on a laptop is the case this encoder handles
worst.

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
  watermark.py      provenance marking of generated audio
  pipeline.py       VoiceLab -- the API the CLI is built on
  evaluate.py       measured self-evaluation
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
would be a guess about your microphone, your room, and your speakers.

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
| `xtts` | text → speech, speech → speech | neural extras, ~2 GB | 17 languages including Arabic; **non-commercial licence** |
| `yourtts` | text → speech | neural extras | lighter, lower fidelity, en/fr/pt |

Switching encoder changes nothing else — the gallery refuses to mix embeddings
from different encoders rather than silently comparing incomparable vectors.

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

180 tests, about 30 seconds, no network access and no downloads. Several of them
pin the claims in this README: if the measured accuracy or the watermark
robustness regresses, the suite fails.

## Licence

MIT (see `LICENSE`). Pretrained models downloaded by the optional backends carry
their own licences — XTTS-v2 in particular is non-commercial.
