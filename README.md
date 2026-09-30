# SnapGuard AI: Edge AI Road Safety Risk Detection

A small local prototype that detects road users (cars, motorcycles, buses,
trucks, pedestrians, bicycles) with a pretrained YOLOv8n model and classifies
a simple scene risk: **SAFE / CAUTION / HIGH RISK**. No cloud API is used for
inference.

## A. Implemented and tested now

Fill in this checklist ONLY after you have actually run each step.

- [ ] `python test_risk.py` passes
- [ ] `python app.py --source sample` produces `outputs/result.jpg`
- [ ] Image input tested with my own image
- [ ] Video input tested
- [ ] Webcam input tested

Tested on: `<machine / CPU / RAM>`  |  OS: `<Windows version>`
Python: `<version, x64 or ARM64>`  |  Date: `<date>`
Measured inference time shown in the on-screen banner: `<your observation, machine-specific>`

**Not tested on Qualcomm/Snapdragon hardware** unless stated here: `<update if true>`

## How it works
1. YOLOv8n (COCO-pretrained, CPU) detects 6 road-related classes.
2. `risk.py` scores each object:
   - Proximity proxy = bounding-box height / frame height (FAR / MID / NEAR).
     Vulnerable users (person, bicycle, motorcycle) use lower thresholds.
   - "In path" = box center inside the middle 40% of the frame width.
3. Scene level = worst object. HIGH RISK: NEAR object in path, or a
   pedestrian/cyclist/motorcycle at MID+ in path. CAUTION: MID vehicle in
   path, NEAR vulnerable user at the side, or 6+ objects.

## Limitations
- Heuristic only: no real distance, speed, depth, or camera calibration.
- Parked/static objects in the center can trigger warnings.
- Thresholds are constants in `risk.py` and need tuning per camera.
- Not a safety-certified system. Demonstration only.

## B. Proposed Snapdragon pathway (NOT implemented or verified by us)
See the "Snapdragon pathway" section of the project notes. Summary:
Qualcomm AI Hub lists a YOLOv8-Detection (YOLOv8-N, 640x640) model with
Snapdragon X Elite / X Plus listed as supported compute targets. Export via
the AI Hub service, then swap the `detector.py` backend. We have not done
this, and we report no Snapdragon latency, FPS, or power numbers.

## Licensing
Ultralytics YOLOv8 is AGPL-3.0. Review the license before distributing.
