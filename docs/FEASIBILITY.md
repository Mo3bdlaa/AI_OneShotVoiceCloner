# How far does one-shot voice cloning actually go?

The short version: **recognising a voice from a few seconds of audio is a solved
problem you can implement yourself. Reproducing a voice from a few seconds is not,
and cannot be implemented from first principles — it requires a model trained on
thousands of hours of speech.**

This document separates the two, because conflating them is the single most common
mistake in projects like this one.

---

## 1. The voice print

### What a voice print is

A vector, a few hundred numbers long, computed from a recording such that two
recordings of the same person land close together and recordings of different
people land far apart. "Close" is measured by cosine similarity.

Nothing about this requires machine learning. A voice differs from another voice
in measurable physical ways:

- **pitch register** — vocal-fold length and tension, roughly 85 Hz to 255 Hz
  across adults, with individual ranges much narrower than that;
- **formant frequencies** — resonances of the vocal tract, set largely by its
  length, which is why children, women and men cluster differently;
- **spectral tilt and breathiness** — how completely the glottis closes;
- **articulation dynamics** — how fast the speaker moves between targets.

The `dsp` encoder measures each of these and concatenates them. See
`voxprint/encoders/dsp.py` for the block-by-block breakdown.

### What makes it work — and what makes it fail

Two things dominate, and neither is obvious in advance.

**Standardisation.** Every human voice shares most of its spectral structure, so
raw feature statistics all point in nearly the same direction. Measured on the
test cast: same-speaker cosine 0.99, different-speaker cosine 0.98. There is no
threshold that separates those. Dividing each dimension by the standard deviation
of a reference population re-weights the space so the dimensions that actually
vary between speakers dominate: same-speaker 0.82–0.89 against different-speaker
below 0.57. This step is worth more than every feature-engineering decision
combined.

**Channel.** About half the blocks describe the absolute spectrum, and a
microphone or a room colours the absolute spectrum. Cepstral mean normalisation
removes most of that for the `mfcc_cmvn` block — a convolutional channel is
additive in the log domain — but the blocks that carry vocal-tract size cannot be
normalised that way without discarding what they measure. The result is an
encoder that is strong within one recording setup and weak across two.

### Measured accuracy

12 enrolled speakers and 12 strangers from one mutually distinct pool, three
4-second enrolment takes each, held-out queries. Reproduce with
`python tools/benchmark.py --encoders dsp ecapa`.

| query | `dsp` rank-1 | `dsp` accepted | `ecapa` rank-1 | `ecapa` accepted |
|---|---|---|---|---|
| clean | 100 % | 91.7 % | 100 % | 100 % |
| 40 dB SNR | 100 % | 2.8 % | 100 % | 41.7 % |
| 30 dB SNR | 86.1 % | 0 % | 77.8 % | 0 % |
| 20 dB SNR | 52.8 % | 0 % | 86.1 % | 0 % |
| 10 dB SNR | 50.0 % | 0 % | 91.7 % | 0 % |

Read the rank-1 and accepted columns as two separate findings.

**Rank-1 is about the encoder.** ECAPA still ranks the right speaker first 91.7 %
of the time at 10 dB SNR, where the DSP encoder has fallen to 50 %. That is the
gap between learning what distinguishes speakers from thousands of labelled
examples and being told what to measure. And ECAPA is understated here: it was
trained on real speech, and vowel-only synthetic voices are outside its domain.

**Acceptance is about the threshold.** It collapses to zero for *both* encoders,
because both were given a threshold from clean-against-clean comparisons and any
condition mismatch shifts every score downward. Changing encoder does not fix
this. `voxprint calibrate --robust` does, by scoring degraded copies of the
enrolment audio during calibration: measured on five speakers, acceptance at
30 dB SNR rises from 50 % to 83 % with clean acceptance unchanged, while the
equal error rate goes from 0.058 to 0.102. That higher error rate is not a
regression -- it is what the system's accuracy always was under mismatch, now
visible instead of hidden behind a threshold nobody could meet.

### A note on the fixture itself

An earlier version of this evaluation drew enrolled speakers and strangers
independently from the voice generator. The generator has only ten identity
parameters, so it happily produced a "stranger" closer to an enrolled speaker
than two enrolled speakers were to each other -- and the reported open-set
rejection then swung between 100 % and 33 % with the seed, measuring the fixture
rather than the system. :func:`voxprint.synthetic.distinct_speakers` now draws
one mutually separated pool and splits it, which is why the numbers above are
stable. Worth recording because the failure looked exactly like a system defect.

Query duration, with enrolment held at 4 seconds:

| query length | accuracy |
|---|---|
| 1 s | 0 % |
| 2 s | 57 % |
| 3 s | 97 % |
| 5 s | 95 % |
| 12 s | 100 % |

(A naive sweep shows a dip at 10 s. That is an artifact of the synthetic voice
generator, which stretches each vowel to fill the requested duration and so
changes the phonetic content; holding segment length fixed and lengthening the
clip gives 100 %. Worth stating because it is exactly the kind of measurement
artifact that gets published as a finding.)

### The honest comparison

A discriminatively trained encoder — ECAPA-TDNN, x-vectors, ResNet-based systems —
reaches roughly 1 % equal error rate on VoxCeleb1, across different recordings,
different microphones, and real background noise. The `dsp` encoder in this
repository does not come close to that on real audio, and no amount of feature
engineering will close the gap, because the neural system learned what varies
*between* speakers from thousands of labelled examples rather than being told.

Use `--encoder ecapa` when accuracy matters. The rest of this project is
unchanged by that choice: the DSP encoder exists so the pipeline is complete,
testable and dependency-free, not because it is the best available option.

---

## 2. Recognising the voice later

This part is genuinely easy once the voice print works, and it is where most of
the engineering value in this repository lies.

**Enrolment** averages several takes. The centroid cancels session noise while the
speaker direction reinforces itself. Keeping the individual takes as well gives a
free quality signal: if a speaker's own takes disagree with each other
(`cohesion` below about 0.55), something is wrong — a second person in the room,
a different microphone, or clipping. That warning is available without any
labelled data and catches most bad enrolments.

**The open-set problem** is the part people forget. Answering "which of these five
people is it?" is easy. Answering "is it any of them?" requires a threshold, and a
threshold is a claim about your data. `voxprint calibrate` derives it: each
enrolment take is scored against the leave-one-out centroid of its own speaker
(scoring it against a centroid it helped compute would inflate the target
distribution and set the threshold far too high) and against every other speaker,
and the equal-error point is found from those two distributions.

For security-shaped uses, calibrate with `--criterion far --max-far 0.001`
instead: fix the rate of impostors you will tolerate and accept whatever miss rate
follows, rather than balancing the two errors as if they cost the same.

One caveat stated plainly: calibration uses enrolment audio, so the reported
false-accept rate is a floor. Real queries come from other people, other rooms and
other microphones, and the true rate is higher. `voxprint eval` on a real corpus
reports a *trial* EER from held-out queries alongside the enrolment-based one, and
the difference between the two is the size of that optimism.

---

## 3. Imitating the voice

Here the two meanings of "cloning" separate, and the difference is not a matter of
degree.

### Voice conversion (speech → speech)

A recording already contains the words, the timing and the prosody. Only the
timbre has to change. That is tractable with signal processing, and `dspvc` does
it: shift the median pitch onto the target's using a phase vocoder, then apply the
difference between the two long-term spectral envelopes as a smooth equalisation
curve.

Measured effect, from the `omar` -> `hana` case in `examples/demo.py` (run it and
you will get these numbers back):

| | similarity to the target's voice print |
|---|---|
| source, unmodified | −0.52 |
| after conversion | +0.19 |
| threshold for a match | +0.51 |

`voxprint.evaluate.conversion_report()` measures a different speaker pair and
shows the same pattern: −0.11 before, +0.37 after, still short of the threshold.

So it closes roughly half the gap and does not come close to passing as the
target.
It cannot: rhythm, accent and the small timing gestures that make a voice
recognisable all belong to the source speaker and stay there, and the formants
move with the pitch, which is audible on large shifts. The honest description is
*timbre transfer*, and the module docstring says so.

It is still worth having: it runs in real time on a CPU with no model download, it
gives the pipeline something to exercise end to end, and it is a fair baseline to
measure a neural backend against.

### Text-to-speech cloning (text → speech)

This is what people mean by "one-shot voice cloning", and it cannot be built from
first principles. Generating a sentence nobody has ever recorded, in a voice heard
for six seconds, requires knowing how voice identity maps onto acoustics across
all phonemes — knowledge that only exists inside a model trained on thousands of
speakers.

XTTS-v2 does this well. It is wired in as the `xtts` backend, it is a ~2 GB
download, and it has been run end to end on this codebase rather than merely
imported. Measured on CPU with `torch 2.8.0`:

| | result |
|---|---|
| English, 44 characters | 3.6 s of 24 kHz audio in 27 s (first call, includes warm-up) |
| Arabic, 48 characters | 4.6 s of audio in 6.6 s — about 1.4× real time once warm |
| conditioning | reference at 228 Hz → output at 230–258 Hz; reference at 143 Hz → output at 160 Hz |
| two runs, one reference | +0.81 against each other |
| two different references | +0.66 |

The pitch row is the decisive one. Given two reference recordings differing only
in register, the model returned speech in the matching register each time — it is
using the reference, not falling back to a default voice. (The similarity scores
are less informative here: identical text inflates both, and the `dsp` encoder is
content-sensitive.)

One caveat about those numbers: the references were the same procedural voices
used everywhere else in this repository, which are vowel-only and unlike real
speech. XTTS handled them, but its fidelity on a real human reference is not
something a synthetic fixture can measure. For that, record yourself.

Three practical things before building on it:

- **Licence.** The XTTS-v2 checkpoint is released under the Coqui Public Model
  License, which does not permit commercial use. The MIT licence on this
  repository covers the source code only. Coqui asks for that agreement on first
  download; this backend refuses with the terms named rather than answering for
  you, so you must set `COQUI_TOS_AGREED=1` yourself.
- **Two dependency pins, both found by hitting them.** `transformers>=5` removed
  `isin_mps_friendly`, which Coqui TTS imports — installation succeeds and the
  import then fails. And `torch>=2.9` drops torchaudio's built-in audio IO for
  `torchcodec`, which needs FFmpeg shared libraries on the system; without them
  the model loads and then dies reading the reference file. Both are pinned in
  `requirements-neural.txt` with the reason recorded.
- **Quality varies by language.** Arabic is supported and works, but the training
  data is far smaller than for English, and it shows — particularly in prosody and
  in the handling of unvowelled text.

Alternatives worth evaluating if XTTS does not fit: OpenVoice (MIT-licensed,
separates tone colour from style), F5-TTS, and Chatterbox. The backend interface
in `voxprint/synth/base.py` is small — a new backend is one class with a
`synthesize` method.

---

## 4. The uncomfortable part

The recogniser and the cloner in this repository point at each other. The same
voice print that verifies someone's identity also conditions a model that
reproduces their voice, and a system that can be fooled by a clone is a system
whose verification claim is worth less than it looks.

That is not hypothetical: `voxprint convert --id <speaker>` prints the output's
similarity to the target's voice print and states whether it would pass. Run it
against a neural backend and you can measure your own recogniser's vulnerability
directly.

The design consequences are in `docs/ETHICS.md`: consent records required at
enrolment, reference audio stored separately and only on request, deletion as easy
as enrolment, and every generated file watermarked by default.

The practical consequence is simpler. **Do not use voice alone to authorise
anything that matters.** Voice recognition is a good signal among several. It is
not a password.
