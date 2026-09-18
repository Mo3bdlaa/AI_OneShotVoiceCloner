"""voxprint -- voice fingerprinting, speaker identification, and voice imitation.

Three capabilities, deliberately kept separate because they differ enormously in
how well they actually work:

1. **Fingerprint** a voice from a short recording (:mod:`voxprint.encoders`).
2. **Recognise** it later, including saying "I don't know this person"
   (:mod:`voxprint.scoring`).
3. **Imitate** it (:mod:`voxprint.synth`) -- either by re-voicing an existing
   recording, which keeps the words, timing and performance and changes only the
   voice, or by generating new speech from text. :mod:`voxprint.song` wraps the
   first for recordings that still have a backing track.

Steps 1 and 2 run anywhere with NumPy and SciPy. Step 3, done properly, needs a
model somebody else trained on thousands of hours of speech; see ``docs/FEASIBILITY.md``
for what each backend can and cannot deliver.

Quick start::

    from voxprint import VoiceLab, ConsentRecord

    lab = VoiceLab("voices")
    lab.enroll("omar", ["omar_1.wav", "omar_2.wav"],
               consent=ConsentRecord(granted_by="Omar", statement="agreed on 2026-01-04"))
    lab.calibrate()
    print(lab.identify_file("unknown.wav").as_dict())
"""

from .audio import TARGET_SR, load_audio, save_audio
from .encoders import SpeakerEncoder, available_encoders, get_encoder
from .gallery import ConsentRecord, Gallery, GalleryError, VoicePrint
from .normalization import EmbeddingStandardizer
from .pipeline import EnrollmentReport, VoiceLab
from .scoring import CalibrationResult, IdentifyResult, VerifyResult, calibrate, identify, verify
from .song import revoice_song, separate
from .synth import available_synths, get_synth
from .watermark import detect_watermark, embed_watermark

__version__ = "0.1.0"

__all__ = [
    "TARGET_SR",
    "CalibrationResult",
    "ConsentRecord",
    "EmbeddingStandardizer",
    "EnrollmentReport",
    "Gallery",
    "GalleryError",
    "IdentifyResult",
    "SpeakerEncoder",
    "VerifyResult",
    "VoiceLab",
    "VoicePrint",
    "available_encoders",
    "available_synths",
    "calibrate",
    "detect_watermark",
    "embed_watermark",
    "get_encoder",
    "get_synth",
    "identify",
    "load_audio",
    "revoice_song",
    "save_audio",
    "separate",
    "verify",
    "__version__",
]
