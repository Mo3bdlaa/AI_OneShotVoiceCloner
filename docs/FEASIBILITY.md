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

### Measured accuracy on real speech

LibriSpeech `dev-clean`: 30 enrolled speakers, 3 enrolment files each, 150
held-out queries, 10 speakers held out entirely as strangers. Reproduce with
`voxprint eval <corpus> --encoder ecapa`.

| encoder | EER (1-to-1) | EER (open-set) | correct | strangers rejected |
|---|---|---|---|---|
| `dsp` | 0.131 | 0.333 | 58 % | 88 % |
| `ecapa` | **0.011** | **0.033** | **100 %** | **100 %** |

This is the number that answers the question. ECAPA's 1.1 % equal error rate is
the VoxCeleb-class figure, reproduced here rather than cited; on this corpus it
identified every held-out query correctly and rejected every stranger. The DSP
encoder gets 58 % — genuinely better than chance across 30 speakers, and
genuinely not good enough for anything that matters.

Two caveats that cut in opposite directions. LibriSpeech is read audiobook speech
recorded cleanly, which is *easier* than conversational audio over a phone. And
the DSP encoder's shipped reference statistics are fitted partly on LibriSpeech
`dev-other`, a different set of speakers but the same corpus and recording style,
so its 58 % is its best case for domain match.

### Measured accuracy on synthetic voices

12 enrolled speakers and 12 strangers from one mutually distinct pool, three
4-second enrolment takes each, held-out queries. Reproduce with
`python tools/benchmark.py`. These are a regression check on the pipeline, not a
claim about people.

| query | `dsp` rank-1 | `dsp` accepted |
|---|---|---|
| clean | 100 % | 91.7 % |
| 40 dB SNR | 100 % | 5.6 % |
| 30 dB SNR | 91.7 % | 0 % |
| 20 dB SNR | 91.7 % | 0 % |
| 10 dB SNR | 75.0 % | 0 % |

Read the rank-1 and accepted columns as two separate findings.

**Rank-1 is about the encoder**, and both encoders hold up: ECAPA still ranks the
right speaker first 92 % of the time at 10 dB SNR.

**Acceptance is about the threshold.** It collapses to zero for *both* encoders,
because both were given a threshold from clean-against-clean comparisons and any
condition mismatch shifts every score downward. Changing encoder does not fix
this. `voxprint calibrate --robust` does, by scoring degraded copies of the
enrolment audio during calibration: measured on five speakers, acceptance at
30 dB SNR rises from 50 % to 83 % with clean acceptance unchanged, while the
equal error rate goes from 0.058 to 0.102. That higher error rate is not a
regression -- it is what the system's accuracy always was under mismatch, now
visible instead of hidden behind a threshold nobody could meet.

### Two thresholds, not one

The other measurement that changed the design. Verification asks "is this the
claimed person?" and compares one score against one claim, so its impostor
distribution is pairwise. Identification asks "which of these N, if any?" and
takes the **maximum** over N — and the maximum of N draws sits far above a single
draw.

On 30 real LibriSpeech speakers: pairwise impostor scores average −0.020, while
an unenrolled speaker's best match over those 30 averages +0.354. A pairwise
equal-error threshold of 0.158 therefore admitted **49 of 50 strangers**, despite
being perfectly well calibrated for the question it was actually answering.

`calibrate` now measures both. The identification threshold comes from a
leave-one-speaker-out best-match distribution: each enrolled speaker is scored
against the gallery *without* themselves, which is exactly the situation a
stranger is in. Switching `identify` to it took stranger rejection from 2 % to
70 % on that corpus, and accuracy from 67 % to 57 % — a real cost, now visible
and adjustable instead of hidden behind a broken default.

The identification threshold depends on how many speakers are enrolled, since the
maximum is over more candidates. Recalibrate after the gallery grows.

### Which reference population to standardise against

Measured on real speech, with everything else held fixed:

| reference population | synthetic margin | real correct | strangers rejected |
|---|---|---|---|
| 400 procedural voices | +0.107 | 56.7 % | 60.0 % |
| 900 LibriSpeech utterances | +0.044 | 56.7 % | 82.0 % |
| both pooled | +0.081 | 58.0 % | 88.0 % |

Neither source alone is right — each is out of domain for the other — and pooling
beats both on their own ground. The shipped reference is the pooled one.

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

This is the easier half of "cloning", and the one that works best. A recording
already contains the words, the timing and the prosody; only the timbre has to
change. There are two very different ways to do it.

**With a trained model.** Zero-shot converters take a few seconds of the target
and re-voice anything. Measured between two LibriSpeech speakers, scoring the
output against the target's held-out ECAPA voice print — the source scored −0.041
to the target and +0.918 to itself before conversion:

| backend | → target | → source | time |
|---|---|---|---|
| `knnvc` | **+0.645** | **+0.031** | 8 s |
| `openvoice` | +0.401 | +0.265 | 6 s |
| `freevc` | +0.239 | +0.161 | 33 s |

kNN-VC wins on both axes: it reaches the target furthest *and* scrubs the source
identity most completely. The mechanism explains why — it replaces each frame's
self-supervised feature with nearest neighbours drawn from the target's own
recordings, so the output is assembled out of the target's actual acoustics
rather than steered by a single summary vector. The same mechanism means more
reference audio keeps helping, where an encoder-based converter saturates:

| reference | 5 s | 10 s | 20 s | 40 s | 49 s |
|---|---|---|---|---|---|
| → target | +0.531 | +0.647 | +0.732 | +0.776 | +0.749 |

Twenty to forty seconds is the useful range. `MAX_REFERENCE_SECONDS` in the
pipeline is 60 for this reason; it was 30 until this was measured.

**The complete loop.** Re-voicing a recording of speaker A toward speaker B with
47 s of B's audio, then asking the system's own recogniser who is speaking:
`MATCH: 1988` at +0.617, with the original speaker A at +0.003. The clone passes
the recogniser. Nothing states the case against voice authentication more
plainly.

**Across languages.** Conversion has no notion of language: there is no text, no
phoneme model, nothing that knows what is being said. The words come from the
input recording, so Arabic in gives Arabic out. That is the theory; measured, it
is *mostly* true. Converting Arabic and English speech toward the same target --
whose reference audio is English only -- and scoring against their held-out voice
print:

| source language | before | after |
|---|---|---|
| English | +0.009 | **+0.631** |
| Arabic | +0.022 | **+0.483** |

Both work. Arabic lands about 0.15 lower, which is the cost of a reference that
never contains the phonemes Arabic has and English does not. The fix is not a
different converter: **use reference audio of the target speaking the language
you will be converting.**

One caveat on that number: ECAPA itself was trained on VoxCeleb, which is
predominantly English, so some of the gap may be the measuring instrument rather
than the conversion. Separating the two would need a speaker encoder trained on
Arabic.

**Without a trained model.** `dspvc` shifts the median pitch onto the target's
using a phase vocoder, then applies the difference between the two long-term
spectral envelopes as a smooth equalisation curve.

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
target — which is exactly the difference between signal processing and a model
that has seen thousands of speakers.
It cannot: rhythm, accent and the small timing gestures that make a voice
recognisable all belong to the source speaker and stay there, and the formants
move with the pitch, which is audible on large shifts. The honest description is
*timbre transfer*, and the module docstring says so.

It is still worth having: it runs in real time on a CPU with no model download, it
gives the pipeline something to exercise end to end, and it is a fair baseline to
measure a neural backend against.

### Singing

Everything above was measured on speech. Singing is a different problem, and the
honest answer is that none of the zero-shot backends here handles it well.

Three reasons, all structural. Every converter was trained on speech, so
sustained vowels, vibrato and a two-octave range are out of domain. A song needs
its vocal separated from the backing track first, and separation leaves artefacts
that conversion amplifies. And melody lives in the pitch contour, which
speech-trained models were never asked to preserve exactly.

The pipeline is in :mod:`voxprint.song` and works mechanically: on a test mix, the
separated vocal correlated +0.968 with the true vocal and +0.064 with the
backing, and the re-voiced result scored +0.668 mixed with the instrumental,
+0.700 as a bare vocal. Those numbers come from speech over a backing track, not
from real singing, and should be read as "the plumbing is correct", not "singing
works".

What actually does singing well — RVC, so-vits-svc — is **not zero-shot**. It
needs roughly ten minutes of the target voice and a training run, which is the
real price of a convincing sung result. `voxprint.song` is the right shape for
those systems too: swap the converter, keep the separation and the remix.

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
