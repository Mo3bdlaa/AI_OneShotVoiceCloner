# Consent, provenance, and what this tool should not be used for

Voice cloning is dual-use in a way that most audio software is not. The same code
path that lets someone give a voice to an assistive device lets someone else
impersonate a relative on the phone. This document describes what the project does
about that, and — as importantly — what it cannot do.

## What the code enforces

**Consent is required at enrolment.** `Gallery.enroll` refuses without a
`ConsentRecord` naming who agreed and to what. The record is stored on the voice
print itself, not in a separate log, so it cannot drift away from the data it
authorises. `--no-consent-check` exists for synthetic voices and test fixtures,
and `voxprint list` prints `NO CONSENT RECORD` in capitals for anything enrolled
that way.

This is a speed bump, not a guarantee. It cannot verify that consent was actually
given. What it does is make the absence of consent a deliberate, visible act
rather than an oversight, and leave an auditable record when it was given.

**Reference audio is separate and optional.** Identification needs only the
embedding; cloning needs real audio. They are stored apart, and `--no-audio`
enrols a speaker who can be recognised but not imitated. An embedding cannot be
played back; a stored recording can.

**Deletion is as easy as enrolment.** `voxprint remove --id <speaker>` erases the
print and the reference audio in one command. If withdrawing consent is harder
than granting it, consent was never meaningful.

**Generated audio is marked by default.** Every file produced by `convert` or
`speak` carries a spread-spectrum watermark, checkable with
`voxprint watermark check`.

## What the watermark is and is not

Measured behaviour, from `tests/test_watermark.py` and the module docstring:

- 34 dB below the signal — inaudible at normal listening levels;
- detected at z ≥ 27 on a 2-second clip, against a peak of 5.8 on unmarked audio;
- survives arbitrary cropping, 20 dB SNR added noise, and band-limiting to 12 kHz;
- **does not** survive resampling through 8 kHz (telephone bandwidth), where only
  2 of 10 test clips still detect.

So: it answers "was this file produced by this tool?" for a cooperative verifier.
It does not answer "is this audio real?" against someone who does not want it to.
Anyone with the key can strip it, and anyone willing to degrade the audio can
destroy it without the key. **A negative result proves nothing.**

No watermarking scheme available today does better against a motivated adversary.
Treat provenance marking as accountability infrastructure for honest users, not as
a detector.

## What the recogniser is not

**Do not use voice alone to authorise anything that matters.** Not payments, not
account recovery, not physical access. Reasons, in order of severity:

1. A good neural cloning backend can produce audio that scores above a threshold
   calibrated on genuine recordings. You can measure this yourself:
   `voxprint convert --id <speaker>` reports the output's similarity to the target
   and whether it would pass.
2. The reported false-accept rate comes from calibration on enrolment audio. Real
   impostors, rooms and microphones push it higher.
3. The default encoder degrades sharply under noise and across microphones — see
   the measured tables in `docs/FEASIBILITY.md`.

Voice recognition is a useful signal among several. It is not a password, and a
system that treats it as one is weaker than one that never claimed to
authenticate at all.

## Legal note, not legal advice

Voice prints are biometric data under several regimes — GDPR Article 9 in the EU,
BIPA in Illinois, and others — which typically require explicit consent, a
retention limit, and a deletion path. Several jurisdictions separately restrict
synthetic voice of a real person regardless of consent, particularly in
advertising, political communication and anything presented as a genuine
recording.

The consent records and deletion path here exist to make compliance possible. They
do not establish it. If you deploy this, talk to someone qualified in your
jurisdiction.

## Uses this project is not built for

Impersonating a specific real person without their knowledge; defeating voice
authentication; generating audio presented as a genuine recording of someone who
did not say it; covert identification of people who have not consented to
enrolment.

None of that is prevented by the code — the code cannot tell intent — but none of
it is what the project is for, and the defaults are set accordingly: consent
required, watermark on, reference audio opt-in, deletion one command away.
