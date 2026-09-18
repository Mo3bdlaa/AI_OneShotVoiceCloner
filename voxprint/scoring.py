"""Matching, open-set decisions, and threshold calibration.

Identification is only half the problem. The hard half is deciding when the best
match is still *not good enough* -- an open-set system must be able to answer
"nobody I know". That decision is a threshold, and a threshold that has not been
calibrated against real scores is a guess. :func:`calibrate` turns the enrolled
data itself into a calibration set, so the number comes from measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .encoders.base import average_embeddings, l2_normalize
from .gallery import Gallery

DEFAULT_THRESHOLD = 0.5
"""Fallback used only when nothing has been calibrated. Deliberately mid-range:
it will be wrong for any particular setup, which is why every code path that
uses it says so."""


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Candidate:
    speaker_id: str
    score: float
    probability: float | None = None
    display_name: str = ""

    def as_dict(self) -> dict:
        return {
            "speaker_id": self.speaker_id,
            "display_name": self.display_name,
            "score": round(self.score, 4),
            "probability": None if self.probability is None else round(self.probability, 4),
        }


@dataclass(frozen=True)
class IdentifyResult:
    """Outcome of a 1-to-N search, including the "unknown" verdict."""

    candidates: list[Candidate]
    threshold: float
    decision: str                       # "match" | "unknown" | "ambiguous" | "empty"
    margin: float = 0.0                 # gap between the top two candidates
    calibrated: bool = True

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def accepted(self) -> bool:
        return self.decision == "match"

    def as_dict(self) -> dict:
        return {
            "decision": self.decision,
            "threshold": round(self.threshold, 4),
            "margin": round(self.margin, 4),
            "calibrated": self.calibrated,
            "candidates": [c.as_dict() for c in self.candidates],
        }


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of a 1-to-1 claim check."""

    speaker_id: str
    score: float
    threshold: float
    accepted: bool
    probability: float | None = None
    calibrated: bool = True

    def as_dict(self) -> dict:
        return {
            "speaker_id": self.speaker_id,
            "score": round(self.score, 4),
            "threshold": round(self.threshold, 4),
            "accepted": self.accepted,
            "probability": None if self.probability is None else round(self.probability, 4),
            "calibrated": self.calibrated,
        }


# --------------------------------------------------------------------------- #
# Core similarity
# --------------------------------------------------------------------------- #

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two embeddings, in ``[-1, 1]``."""
    return float(np.dot(l2_normalize(np.asarray(a, dtype=np.float64)), l2_normalize(np.asarray(b, dtype=np.float64))))


def identify(
    query: np.ndarray,
    gallery: Gallery,
    *,
    threshold: float | None = None,
    top_k: int = 5,
    min_margin: float = 0.0,
    scaler: PlattScaler | None = None,
) -> IdentifyResult:
    """Search the gallery for the speaker of ``query``.

    ``min_margin`` guards against the case where two enrolled speakers both score
    above threshold and are nearly tied: confidently naming either would be
    wrong, so the result is reported as ambiguous instead.
    """
    calibrated = threshold is not None or gallery.threshold is not None
    thr = threshold if threshold is not None else (gallery.threshold or DEFAULT_THRESHOLD)

    ids, centroids = gallery.centroid_matrix(standardize=True)
    if not ids:
        return IdentifyResult([], thr, "empty", 0.0, calibrated)

    q = gallery.standardizer.transform(np.asarray(query, dtype=np.float64))
    scores = centroids @ q

    order = np.argsort(-scores)[: max(1, top_k)]
    candidates = [
        Candidate(
            speaker_id=ids[i],
            score=float(scores[i]),
            probability=scaler.probability(float(scores[i])) if scaler else None,
            display_name=gallery.prints[ids[i]].display_name,
        )
        for i in order
    ]

    margin = float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else float("inf")
    if candidates[0].score < thr:
        decision = "unknown"
    elif margin < min_margin:
        decision = "ambiguous"
    else:
        decision = "match"
    return IdentifyResult(candidates, thr, decision, 0.0 if np.isinf(margin) else margin, calibrated)


def verify(
    query: np.ndarray,
    gallery: Gallery,
    speaker_id: str,
    *,
    threshold: float | None = None,
    scaler: PlattScaler | None = None,
) -> VerifyResult:
    """Check a claimed identity (1-to-1)."""
    calibrated = threshold is not None or gallery.threshold is not None
    thr = threshold if threshold is not None else (gallery.threshold or DEFAULT_THRESHOLD)
    print_ = gallery.get(speaker_id)

    q = gallery.standardizer.transform(np.asarray(query, dtype=np.float64))
    ref = gallery.standardizer.transform(print_.centroid)
    score = float(np.dot(q, ref))
    return VerifyResult(
        speaker_id=speaker_id,
        score=score,
        threshold=thr,
        accepted=score >= thr,
        probability=scaler.probability(score) if scaler else None,
        calibrated=calibrated,
    )


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #

@dataclass
class CalibrationResult:
    """Threshold plus the measurements that justify it."""

    threshold: float
    eer: float
    n_target: int
    n_impostor: int
    target_scores: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    impostor_scores: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    criterion: str = "eer"
    #: Problems that make the threshold untrustworthy, e.g. no same-speaker
    #: trials because every speaker has a single enrolment take.
    warnings: list[str] = field(default_factory=list)
    #: Acoustic conditions represented in the trials. Empty means clean-only,
    #: which makes the threshold valid only for queries recorded like the
    #: enrolment audio was.
    conditions: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """Whether this calibration actually measured anything.

        With one enrolment take per speaker there are no leave-one-out
        same-speaker trials, so there is nothing to balance against the impostor
        scores and the "threshold" is just the fallback constant. Reporting that
        as calibrated would be a lie that propagates into every later decision.
        """
        return self.n_target > 0 and self.n_impostor > 0 and np.isfinite(self.eer)

    def rates_at(self, threshold: float) -> tuple[float, float]:
        """``(false_accept_rate, false_reject_rate)`` at a given threshold."""
        far = float(np.mean(self.impostor_scores >= threshold)) if self.n_impostor else float("nan")
        frr = float(np.mean(self.target_scores < threshold)) if self.n_target else float("nan")
        return far, frr

    def as_dict(self) -> dict:
        far, frr = self.rates_at(self.threshold)
        return {
            "criterion": self.criterion,
            "usable": self.usable,
            "warnings": list(self.warnings),
            "threshold": round(self.threshold, 4),
            # NaN is not valid JSON -- Python emits a bare `NaN` token that
            # JSON.parse and most strict parsers reject, so a browser client
            # would fail on the whole response rather than on one field.
            "eer": _finite(self.eer),
            "far_at_threshold": None if np.isnan(far) else round(far, 4),
            "frr_at_threshold": None if np.isnan(frr) else round(frr, 4),
            "target_pairs": self.n_target,
            "impostor_pairs": self.n_impostor,
            "target_score_mean": round(float(self.target_scores.mean()), 4) if self.n_target else None,
            "impostor_score_mean": round(float(self.impostor_scores.mean()), 4) if self.n_impostor else None,
            "conditions": list(self.conditions),
        }


def _finite(value: float, digits: int = 4) -> float | None:
    """Round for output, mapping NaN and infinities to ``None`` (JSON ``null``)."""
    return round(float(value), digits) if np.isfinite(value) else None


def collect_trial_scores(gallery: Gallery) -> tuple[np.ndarray, np.ndarray]:
    """Build target and impostor score sets from the gallery itself.

    Target scores use leave-one-out: each utterance is scored against the
    centroid of that speaker's *remaining* utterances. Scoring it against a
    centroid it helped compute would inflate the target distribution and push the
    calibrated threshold far too high.
    """
    labels, vectors = gallery.all_utterances(standardize=True)
    if not labels:
        return np.zeros(0), np.zeros(0)

    by_speaker: dict[str, list[int]] = {}
    for idx, sid in enumerate(labels):
        by_speaker.setdefault(sid, []).append(idx)

    target: list[float] = []
    impostor: list[float] = []

    for sid, idxs in by_speaker.items():
        others = {o: np.stack([vectors[i] for i in oi]) for o, oi in by_speaker.items() if o != sid}
        for i in idxs:
            rest = [j for j in idxs if j != i]
            if rest:
                loo_centroid = average_embeddings(np.stack([vectors[j] for j in rest]))
                target.append(float(np.dot(vectors[i], loo_centroid)))
            for mat in others.values():
                impostor.append(float(np.dot(mat, vectors[i]).mean()))

    return np.asarray(target), np.asarray(impostor)


def equal_error_rate(target: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """Return ``(eer, threshold)`` where false-accept and false-reject rates meet."""
    target = np.asarray(target, dtype=np.float64)
    impostor = np.asarray(impostor, dtype=np.float64)
    if target.size == 0 or impostor.size == 0:
        return float("nan"), DEFAULT_THRESHOLD

    candidates = np.unique(np.concatenate([target, impostor]))
    # Evaluate just below each observed score so a threshold exactly equal to a
    # score behaves consistently with the >= comparison used at decision time.
    grid = np.concatenate([[candidates[0] - 1e-6], candidates, [candidates[-1] + 1e-6]])
    far = np.array([np.mean(impostor >= t) for t in grid])
    frr = np.array([np.mean(target < t) for t in grid])

    diff = far - frr
    idx = int(np.argmin(np.abs(diff)))
    eer = float((far[idx] + frr[idx]) / 2.0)
    return eer, float(grid[idx])


def threshold_for_far(target: np.ndarray, impostor: np.ndarray, max_far: float) -> float:
    """Lowest threshold whose false-accept rate stays at or below ``max_far``.

    This is the knob to reach for in a security setting: fix the rate of
    impostors you are willing to let through, and accept whatever miss rate that
    implies, rather than the other way round.
    """
    impostor = np.asarray(impostor, dtype=np.float64)
    if impostor.size == 0:
        return DEFAULT_THRESHOLD
    grid = np.unique(np.concatenate([impostor, np.asarray(target, dtype=np.float64)]))
    for t in grid:
        if float(np.mean(impostor >= t)) <= max_far:
            return float(t)
    return float(grid[-1] + 1e-6)


def calibrate(gallery: Gallery, *, criterion: str = "eer", max_far: float = 0.01) -> CalibrationResult:
    """Measure the gallery's own score distributions and pick a threshold.

    Caveat worth stating plainly: this calibrates against *enrolled* speakers and
    *enrolment* recordings. Real queries come from other people and other rooms,
    so the true false-accept rate in deployment is higher than the number
    reported here. Treat it as a floor, not a promise.
    """
    target, impostor = collect_trial_scores(gallery)
    eer, eer_threshold = equal_error_rate(target, impostor)
    if criterion == "eer":
        threshold = eer_threshold
    elif criterion == "far":
        threshold = threshold_for_far(target, impostor, max_far)
    else:
        raise ValueError(f"unknown calibration criterion {criterion!r}; use 'eer' or 'far'")

    warnings: list[str] = []
    if target.size == 0:
        single = [p.speaker_id for p in gallery if p.n_utterances < 2]
        who = f"only one take: {', '.join(single)}" if single else "only one take each"
        warnings.append(
            "no same-speaker trials. Leave-one-out scoring needs at least two enrolment takes per "
            f"speaker, and these have {who}. The threshold reported below is the uncalibrated "
            "default -- add another recording per speaker and calibrate again."
        )
    if gallery.spec.dim and len(gallery.prints) < 2:
        warnings.append("fewer than two speakers: there are no impostor trials to measure against")

    return CalibrationResult(
        threshold=float(threshold),
        eer=eer,
        n_target=int(target.size),
        n_impostor=int(impostor.size),
        target_scores=target,
        impostor_scores=impostor,
        criterion=criterion,
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# Score -> probability
# --------------------------------------------------------------------------- #

@dataclass
class PlattScaler:
    """Logistic map from a similarity score to P(same speaker).

    A cosine of 0.7 means nothing on its own; "84% likely the same speaker, given
    how this gallery scores" is something a caller can act on. Fitted by
    Newton-Raphson on the target/impostor scores from :func:`calibrate`.
    """

    a: float = 1.0
    b: float = 0.0

    @classmethod
    def fit(cls, target: np.ndarray, impostor: np.ndarray, iterations: int = 100) -> PlattScaler:
        x = np.concatenate([np.asarray(target, dtype=np.float64), np.asarray(impostor, dtype=np.float64)])
        y = np.concatenate([np.ones(np.size(target)), np.zeros(np.size(impostor))])
        if x.size == 0 or len(np.unique(y)) < 2:
            return cls()

        a, b = 1.0, 0.0
        for _ in range(iterations):
            z = a * x + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
            w = np.maximum(p * (1.0 - p), 1e-9)
            err = p - y
            # Gradient and Hessian of the log-likelihood in (a, b).
            g = np.array([np.dot(err, x), err.sum()])
            h = np.array(
                [
                    [np.dot(w * x, x), np.dot(w, x)],
                    [np.dot(w, x), w.sum()],
                ]
            )
            try:
                step = np.linalg.solve(h + 1e-9 * np.eye(2), g)
            except np.linalg.LinAlgError:  # pragma: no cover - degenerate input
                break
            a, b = a - step[0], b - step[1]
            if np.max(np.abs(step)) < 1e-9:
                break
        return cls(a=float(a), b=float(b))

    def probability(self, score: float) -> float:
        z = float(np.clip(self.a * score + self.b, -60, 60))
        return float(1.0 / (1.0 + np.exp(-z)))

    def as_dict(self) -> dict:
        return {"a": round(self.a, 6), "b": round(self.b, 6)}
