# Singing, so-vits-svc, and live conversion

Two questions this document answers: can a song be re-voiced convincingly, and
can the conversion run live into a virtual microphone. Both are yes, with
conditions worth understanding before you spend an afternoon on either.

## Can you skip training entirely?

For speech, yes — and you should. For singing, no, and the reason is measurable.

Singing lives in the pitch contour, so the question is not "does it sound like
the target" but "does the tune survive". Rendering a 15-note melody, converting
it with each zero-shot backend, and comparing the output's F0 contour against the
input's (semitone error is the median absolute deviation after removing the
constant transposition -- a melody moved up an octave is still the melody):

| backend | melody correlation | semitone error | identity reached |
|---|---|---|---|
| the source itself | 1.000 | 0.00 | — |
| `knnvc` | 0.061 | 3.28 | **+0.70** |
| `freevc` | −0.336 | 2.13 | +0.24 |
| `openvoice` | −0.165 | 10.87 | +0.40 |
| **`dspvc`** | 0.281 | **0.23** | weak |

The neural zero-shot converters do not carry a tune. kNN-VC's contour is
uncorrelated with the input's and off by over three semitones; FreeVC and
OpenVoice are *negatively* correlated. They re-voice each frame from the target's
own acoustics, and the pitch comes with it.

The one that does preserve the melody is the least sophisticated: `dspvc` shifts
F0 by a constant ratio and leaves the contour alone, so its semitone error is
0.23 -- an order of magnitude better than any neural backend. What it cannot do
is reach the target's identity.

So, zero-shot, **you can have the identity or the melody, not both**:

| | speech | singing |
|---|---|---|
| no training | `knnvc` — the best option there is | identity without the tune, or the tune without the identity |
| with training | not worth it | `sovits` — the only one that gives both |

(One caveat on the table: the correlation column is noisy — it is sensitive to a
few outlier frames, which is why `dspvc` scores low on it while its median
semitone error says the contour is intact. Read the semitone column.)

## Why the zero-shot converters are not enough

`voxprint revoice` with `knnvc` moves speech to a target voice well — measured at
+0.645 against the target's voice print where the unmodified source scored −0.041
(`docs/FEASIBILITY.md` has the table). On singing it degrades, for three reasons
that are structural rather than fixable by tuning:

1. **They were trained on speech.** Sustained vowels, vibrato, and a two-octave
   range are outside what they have seen.
2. **They do not treat pitch as something to preserve.** Melody lives entirely in
   the F0 contour. A converter that was never asked to keep it exactly will not.
3. **Separation artefacts compound.** A song must have its vocal extracted first,
   and whatever the separator leaves behind, the converter amplifies.

so-vits-svc addresses the first two directly: it conditions on F0 explicitly, and
it is routinely trained on sung material. The price is that it is **not
zero-shot** — and that price is real.

## What 6.5 hours of training actually buys

Trained on 8.2 minutes of one speaker, CPU-only, converting a clip of a different
speaker and scoring the output against the target's **held-out** recordings:

| training steps | → target | → source | wall clock |
|---|---|---|---|
| 0 (unconverted) | −0.024 | +0.931 | — |
| 175 | +0.342 | +0.081 | 8 min |
| 525 | +0.420 | +0.112 | 26 min |
| 1 400 | +0.424 | +0.027 | 71 min |
| **2 800** | **+0.549** | +0.032 | **155 min** |
| 5 075 | +0.427 | +0.089 | ~4 h |
| 5 950 | +0.446 | +0.029 | ~5 h |
| 8 050 | +0.383 | +0.084 | ~6.5 h |
| *kNN-VC, zero-shot* | *+0.700* | *+0.059* | *0* |

Three findings, none of them what "train it longer" suggests.

**The curve is not monotonic.** It peaks at 2 800 steps and then falls back and
oscillates between 0.38 and 0.45 for the next 5 000. Taking the final checkpoint
would have given +0.383 — worse than one taken four hours earlier. Pick the
checkpoint by measuring, not by taking the last one. `tools/svc_curve.py` exists
to make that easy.

**More steps stopped being the constraint.** Upstream asks for 10 000+ steps, and
at 8 050 this run is nowhere near its peak, let alone climbing. The limit is the
8.2 minutes of training audio — below the ten minutes upstream asks for — not the
amount of compute spent on it.

**It never caught the zero-shot converter.** kNN-VC scored +0.700 on the same
clip with no training whatsoever. Even at its best, so-vits-svc reached +0.549.

One caveat that may cap the whole experiment: the training audio was 16 kHz
upsampled to the 44.1 kHz the model expects, so there is no genuine
high-frequency content for it to learn. A speaker encoder keys partly on
full-band timbre, so a run on real 44.1 kHz recordings may have a higher ceiling
than this one did.

**So do not reach for so-vits-svc to convert speech.** kNN-VC beats it for free.
Reach for it when the material is sung, where F0 conditioning is the thing that
matters and no zero-shot converter here will do — and then give it ten or more
minutes of real full-bandwidth audio, a GPU, and a checkpoint chosen by
measurement.

## What so-vits-svc costs

| | |
|---|---|
| training audio | ~10 minutes of clean solo recordings of the target voice |
| training hardware | a GPU; see below for what CPU actually means |
| output | a checkpoint that *is* that voice, reusable indefinitely |
| per-conversion | no reference audio needed — the model is the voice |

Measured on this 4-core CPU box, with 8.2 minutes of training audio:

| step | time |
|---|---|
| resample + config + HuBERT/F0 extraction | 54 s |
| training | upstream asks for 10 000+ steps; hours on a GPU, days here |

One trap worth knowing: a checkpoint is only written every `eval_interval` steps,
which defaults to 200. A short run that finishes without reaching it saves
nothing and looks exactly like a failure. `voxprint train-svc --epochs N` leaves
that default alone; lower it in the config if you are doing a short verification
run.

Preprocessing is cheap. Training is not: upstream's guidance of 10 000+ steps is
hours on a mid-range GPU and days on this CPU. Treat CPU training as a way to
verify the pipeline runs, not to produce a voice you would use.

## Using it

```bash
pip install so-vits-svc-fork

# the speaker must already be enrolled, with a consent record: a trained model
# can generate unlimited audio in their voice, indefinitely
voxprint enroll me_*.wav --id me --consent "my own voice, recorded 2026-01-04"

# train. --prepare-only stops after preprocessing so you can inspect the config
voxprint train-svc --id me --audio-dir ~/recordings/me --epochs 200

# convert a song: the vocal is separated, converted, and mixed back
voxprint revoice song.mp3 --id me --backend sovits -o out.wav

# shift the key when the source and your range do not match
voxprint revoice song.mp3 --id me --backend sovits --transpose -3 -o out.wav
```

### The one setting that matters most

`--auto-predict-f0` is **off** by default here, and it should stay off for
singing. With it on, the model predicts its own pitch contour instead of
following the source — which sounds more natural for speech, and destroys the
melody of a song. Upstream's own help text says the same about realtime use.

`--transpose` shifts the key in semitones. If the song sits outside the range the
model was trained on, no amount of conversion quality substitutes for moving it
there first.

## A NumPy incompatibility, patched

so-vits-svc-fork 4.2.x renders its TensorBoard previews with
`np.fromstring(fig.canvas.tostring_argb(), ...)`. NumPy removed the binary mode
of `fromstring`, so training crashes at its first logging step:

    ValueError: The binary mode of fromstring is removed, use frombuffer instead

It is a logging path, not a modelling one, so `voxprint/_svc_compat.py` patches
it rather than pinning NumPy back for the whole project, and `voxprint.svc` runs
the upstream CLI through that shim. Delete the module once upstream ships a fix.

## Live conversion into a virtual microphone

so-vits-svc ships `svc vc`, which reads a microphone, converts continuously, and
writes to an output device. Making that output appear *as a microphone* to Discord,
Zoom or a browser is an operating-system job — no Python package can do it:

| OS | virtual cable |
|---|---|
| Linux | PulseAudio/PipeWire `module-null-sink` + `module-remap-source` |
| macOS | BlackHole (`brew install blackhole-2ch`) or Loopback |
| Windows | VB-Audio Virtual Cable or VoiceMeeter |

```bash
voxprint realtime --id me
```

prints exactly what is missing, lists your audio devices with their indices, and
gives the per-platform routing commands and the launch line once everything is in
place.

### Latency, honestly

End-to-end delay is the audio block size plus inference time per block. The
default block here is 0.35 s, lowered from upstream's 0.5 s because latency is
the first thing anyone notices; raise it if the audio breaks up. Add
`--passthrough-original` to `svc vc` to hear the buffering delay alone, with no
model in the path — that is the floor you are working against.

**On CPU this is borderline, not hopeless.** Measured here: converting a 10.8 s
clip took 12 s, about 1.1× real time. Live conversion needs inference *faster*
than real time with headroom for the F0 extraction and buffering on top, so this
CPU would fall behind — but not by much. A faster CPU may manage; a GPU
comfortably will. Note that this is the opposite of training, where the CPU is
hopeless by orders of magnitude: inference is cheap, training is not.

### Not verified here

The realtime path is the one part of this project that has not been run
end to end. This container has no PortAudio and no audio devices, so
`voxprint realtime` correctly reports it cannot start, and that failure path is
tested — but the audio path itself, the latency figures, and the virtual-cable
routing are from upstream's interface and documentation, not from a measurement
made here. Everything else in this repository carries numbers that were measured;
this section does not, and says so.

## Consent

A trained voice model is a heavier artefact than an embedding: it generates
unlimited new audio in a person's voice, for as long as the file exists.
`voxprint train-svc` therefore refuses unless the speaker is enrolled with a
consent record. Treat the checkpoint with the same care as the recordings it came
from, and see `docs/ETHICS.md`.
