"""Heuristic road-safety risk logic. Pure Python, no ML dependencies.

This is NOT a calibrated distance estimate. Proximity is approximated by how
tall a bounding box is relative to the frame, and "in path" means the box
center is in the middle of the frame. Tune the constants for your camera.
"""
from dataclasses import dataclass
from typing import List

SAFE, CAUTION, HIGH = "SAFE", "CAUTION", "HIGH RISK"
_RANK = {SAFE: 0, CAUTION: 1, HIGH: 2}

VULNERABLE = {"person", "bicycle", "motorcycle"}
CORRIDOR = (0.30, 0.70)        # horizontal "ego path" as fraction of frame width
NEAR_H_VEHICLE = 0.45          # box height / frame height => NEAR
NEAR_H_VULNERABLE = 0.35       # lower threshold: vulnerable users
MID_H = 0.20                   # => MID
CROWD_COUNT = 6                # this many detections => at least CAUTION


@dataclass
class Detection:
    label: str
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class ScoredObject:
    det: Detection
    proximity: str      # FAR | MID | NEAR
    in_corridor: bool
    severity: str


@dataclass
class RiskResult:
    level: str
    reasons: List[str]
    objects: List[ScoredObject]


def _score(det: Detection, w: int, h: int):
    h_ratio = (det.y2 - det.y1) / h
    cx = ((det.x1 + det.x2) / 2) / w
    in_corr = CORRIDOR[0] <= cx <= CORRIDOR[1]
    vuln = det.label in VULNERABLE
    near_thr = NEAR_H_VULNERABLE if vuln else NEAR_H_VEHICLE

    if h_ratio >= near_thr:
        prox = "NEAR"
    elif h_ratio >= MID_H:
        prox = "MID"
    else:
        prox = "FAR"

    if in_corr and (prox == "NEAR" or (vuln and prox == "MID")):
        sev = HIGH
    elif in_corr and prox == "MID":
        sev = CAUTION
    elif vuln and prox == "NEAR":
        sev = CAUTION
    else:
        sev = SAFE
    return prox, in_corr, sev


def assess(detections: List[Detection], frame_w: int, frame_h: int) -> RiskResult:
    objects: List[ScoredObject] = []
    for d in detections:
        prox, in_corr, sev = _score(d, frame_w, frame_h)
        objects.append(ScoredObject(d, prox, in_corr, sev))

    level = SAFE
    for o in objects:
        if _RANK[o.severity] > _RANK[level]:
            level = o.severity

    flagged = sorted(
        (o for o in objects if o.severity != SAFE),
        key=lambda o: _RANK[o.severity],
        reverse=True,
    )
    reasons: List[str] = []
    for o in flagged:
        where = "in path" if o.in_corridor else "to the side"
        text = f"{o.det.label} {o.proximity.lower()}, {where}"
        if text not in reasons:
            reasons.append(text)
    reasons = reasons[:3]

    if level == SAFE and len(detections) >= CROWD_COUNT:
        level = CAUTION
        reasons.append(f"crowded scene ({len(detections)} objects)")

    if not reasons:
        reasons = ["No close objects in path"]
    return RiskResult(level, reasons, objects)
