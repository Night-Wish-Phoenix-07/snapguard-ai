"""Run with: python test_risk.py  (no pytest needed)."""
from risk import CAUTION, HIGH, SAFE, Detection, assess

W, H = 640, 480


def D(label, x1, y1, x2, y2):
    return Detection(label, 0.9, x1, y1, x2, y2)


def level(*dets):
    return assess(list(dets), W, H).level


assert level() == SAFE, "empty scene"
assert level(D("car", 300, 100, 340, 400)) == HIGH, "big car in path"
assert level(D("car", 300, 200, 340, 240)) == SAFE, "far car in path"
assert level(D("car", 300, 100, 340, 220)) == CAUTION, "mid car in path"
assert level(D("person", 300, 100, 340, 200)) == HIGH, "mid pedestrian in path"
assert level(D("person", 50, 100, 90, 290)) == CAUTION, "near pedestrian at side"
assert level(D("person", 50, 100, 90, 200)) == SAFE, "mid pedestrian at side"
crowd = [D("car", 10 + i * 5, 200, 30 + i * 5, 240) for i in range(6)]
assert level(*crowd) == CAUTION, "crowded scene"
print("All risk tests passed.")
