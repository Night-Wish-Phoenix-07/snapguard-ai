"""Draws boxes, the in-path corridor, and a risk banner (OpenCV, BGR colors)."""
import cv2

from risk import CAUTION, CORRIDOR, HIGH, SAFE, RiskResult

FONT = cv2.FONT_HERSHEY_SIMPLEX
LEVEL_COLORS = {SAFE: (0, 150, 0), CAUTION: (0, 140, 255), HIGH: (0, 0, 210)}
PROX_COLORS = {"FAR": (0, 190, 0), "MID": (0, 165, 255), "NEAR": (0, 0, 255)}


def draw(frame, result: RiskResult, infer_ms: float):
    out = frame.copy()
    h, w = out.shape[:2]
    s = max(0.6, w / 1100)  # simple text scaling for larger frames

    # In-path corridor guide lines
    for frac in CORRIDOR:
        x = int(frac * w)
        cv2.line(out, (x, 0), (x, h), (255, 255, 255), 1)

    # Boxes + labels
    for o in result.objects:
        d = o.det
        color = PROX_COLORS[o.proximity]
        cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), color, 2)
        text = f"{d.label} {d.conf:.2f}"
        (tw, th), base = cv2.getTextSize(text, FONT, 0.5 * s, 1)
        top = max(d.y1 - th - base - 4, 0)
        cv2.rectangle(out, (d.x1, top), (d.x1 + tw + 6, top + th + base + 4), color, -1)
        cv2.putText(out, text, (d.x1 + 3, top + th + 1), FONT, 0.5 * s,
                    (255, 255, 255), 1, cv2.LINE_AA)

    # Banner (two lines)
    banner_h = int(62 * s)
    cv2.rectangle(out, (0, 0), (w, banner_h), LEVEL_COLORS[result.level], -1)
    cv2.putText(out, f"{result.level}  |  {len(result.objects)} objects  |  {infer_ms:.0f} ms",
                (10, int(24 * s)), FONT, 0.7 * s, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, "; ".join(result.reasons[:2]),
                (10, int(50 * s)), FONT, 0.5 * s, (255, 255, 255), 1, cv2.LINE_AA)
    return out
