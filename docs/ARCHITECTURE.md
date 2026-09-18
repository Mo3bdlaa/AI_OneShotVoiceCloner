# Architecture

## The shape of it

```
audio file
    |
    v
audio.py            decode -> mono -> 16 kHz -> trim silence -> VAD -> loudness
    |
    v
features.py         MFCC + deltas | LPC cepstrum | pitch track | spectral shape
    |
    v
encoders/           -> one L2-normalised vector (the voice print)
    |
    v
normalization.py    per-dimension standardisation against a reference population
    |
    +--> gallery.py     store, with consent and quality metadata
    |
    +--> scoring.py     identify / verify / calibrate a threshold
    |
    +--> synth/         imitate, using the stored reference audio
```

Each layer only knows about the one below it. `encoders` takes arrays and returns
vectors; it has no idea files exist. `gallery` takes vectors; it has no idea audio
exists. Everything that touches the filesystem lives in `pipeline.py`, and the CLI
is a shell over that. The practical benefit is that the test suite can exercise
the scoring logic on synthetic vectors in milliseconds, without generating audio.

## Layer notes

### `audio.py` — the front-end contract

One function, `preprocess`, is called on both the enrolment and the query path.
That is not a convenience: any asymmetry between the two shows up as a systematic
score shift, which then looks like a threshold problem and gets "fixed" in the
wrong place.

Loudness normalisation matters more than it looks. Without it, recording gain
becomes a feature the encoder can latch onto, and the same person recorded twice
at different levels scores as two people.

### `features.py` — deliberately dependency-free

Mel filterbank, MFCC, deltas, LPC via Levinson–Durbin, an autocorrelation pitch
tracker, and five spectral-shape descriptors — all on NumPy. The neural encoders
bring their own feature extraction, so this exists purely so the default path has
no heavy dependencies.

The pitch tracker is separate from the cepstral features on purpose. Pitch is the
most speaker-discriminative scalar available and the one a spectral-envelope
encoder systematically under-weights, so it is estimated directly and given its
own block.

### `encoders/` — the plug point

`SpeakerEncoder` is the only interface the rest of the system knows:
`embed(wav, sr) -> unit vector`. Every encoder also publishes an `EncoderSpec`
(name, version, dimension, sample rate) which is stored with each gallery. Loading
a gallery with a different encoder is refused rather than silently comparing
vectors that live in different spaces — a failure mode that would otherwise
produce plausible-looking nonsense.

Heavy backends import their dependencies inside the factory, so
`import voxprint` never pulls in PyTorch.

### `normalization.py` — the step that makes it work

Documented at length in the module and in `docs/FEASIBILITY.md`. The short version:
without it, all cosine scores sit within 0.01 of each other and no threshold
separates anything.

The shipped reference statistics are estimated from procedurally generated voices,
which is a prior rather than ground truth. `Gallery.refit_standardizer()` replaces
them with statistics from the real enrolled population once there are at least
eight speakers and twenty-four takes — below that, a 373-dimensional variance
estimate mostly describes the handful of people it was fitted on, and measurably
weakens held-out scores.

### `gallery.py` — storage with opinions

A `VoicePrint` keeps both the centroid and the individual takes. The centroid is
what you match against; the spread of the takes tells you whether to trust it
(`cohesion`). Consent lives on the print rather than in a separate log, so it
cannot drift away from the data it authorises.

Persistence is a single compressed `.npz` with a JSON manifest, written to a
temporary file and renamed into place. The temporary file's suffix must be `.npz`
— NumPy appends that extension to any other name, and the rename would then point
at an empty stub. That bug existed here; the test `test_save_is_atomic_on_failure`
is what keeps it fixed.

### `scoring.py` — decisions, not just scores

Identification returns one of four verdicts: `match`, `unknown`, `ambiguous`
(two enrolled speakers too close to call), or `empty`. Results carry a
`calibrated` flag, and every code path that falls back to the default threshold
says so in its output — an uncalibrated threshold is a guess, and guesses should
be visible.

Calibration scores each take against the leave-one-out centroid of its own
speaker. Scoring against a centroid the take helped compute inflates the target
distribution and sets the threshold far too high.

### `augment.py` — degradations, for calibration

Noise (white and pink), a synthetic room impulse response, telephone bandwidth, a
microphone tilt, and clipping. Its reason for existing is in the module docstring:
a threshold calibrated on clean audio accepts almost nothing once the query
conditions change, for *both* the DSP and the neural encoder, and no choice of
encoder fixes that. `calibrate --robust` scores degraded copies of the enrolment
audio so the threshold accounts for the shift.

### `corpus.py` — evaluation on real speech

The same protocol as `evaluate.py`, on a folder-per-speaker corpus, with three
things the procedural evaluation cannot give: whole speakers held out as
strangers, a trial EER from held-out queries alongside the optimistic
enrolment-based one, and ranking accuracy reported separately from acceptance.

### `server.py` — HTTP API and browser UI

`http.server` rather than a web framework, so the core install stays at three
dependencies. Binds to `127.0.0.1` by default and says so loudly when told to
bind elsewhere; a voice-print gallery is biometric data and there is no
authentication. Every endpoint maps to a `VoiceLab` method — no logic lives here.

### `synth/` — two capabilities, not one

Backends declare `capabilities` — `{"tts"}`, `{"vc"}`, or both — so asking a
conversion-only backend to read a sentence produces a clear message instead of a
stack trace. `pipeline` checks `available()` before loading anything, so a missing
optional dependency surfaces as one line rather than an import traceback.

### `watermark.py` — a measured design

The first version modulated a 32-bit payload one bit per block. It could not be
combined coherently across blocks and never separated marked from unmarked audio
at an inaudible strength — measured z of 1.0 against 0.87 for clean audio. Two
changes fixed it: fold the tag into the chip sequence so every block carries the
same polarity and their correlations add linearly, and whiten both signal and
chips before correlating so that speech's low-frequency energy stops dominating
the interference. That moved a marked 3-second clip from z = 2.0 to z = 14.4 with
the unmarked peak unchanged at 3.5.

Both dead ends are recorded in the module docstring, because the reasoning is the
part that is hard to reconstruct later.

## Testing strategy

All test audio comes from `voxprint/synthetic.py`: a source-filter model where a
speaker is a pitch register plus a formant set. This makes the suite
deterministic, download-free, and free of recordings of real people.

The suite is built around properties rather than golden outputs:

- the same speaker must score above every impostor (`test_encoder.py`);
- identity must survive a change of spoken content, and of loudness;
- conversion must move the voice print toward the target *and* must not reach it;
- the watermark must be detected after cropping and noise, and must **not** be
  detected after telephone-bandwidth resampling — a documented limitation pinned
  by a test so it cannot be quietly over-claimed later;
- robust calibration must lower the threshold and must not lower clean acceptance;
- `voxprint selftest` numbers must stay within the ranges quoted in the README;
- the fixture itself is tested: `distinct_speakers` must return voices that are
  actually distinct, because an earlier version did not and its open-set numbers
  swung with the seed.

`tests/test_neural_backends.py` covers the optional backends and skips when they
are absent. The XTTS tests additionally need `VOXPRINT_TEST_XTTS=1`, since they
pull a 2 GB checkpoint. They have been run: ECAPA and XTTS both work, including
Arabic synthesis, and the numbers are in `docs/FEASIBILITY.md`.

Synthetic voices are easier than real ones. The suite proves the pipeline is
correct and that the documented claims hold; it does not prove the encoder is
accurate on real speech, and nothing in it is presented as if it did.
