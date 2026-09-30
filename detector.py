"""Thin wrapper around a pretrained YOLOv8n model (COCO), CPU inference."""
from typing import List

from ultralytics import YOLO

from risk import Detection

# COCO class ids -> label
ROAD_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class RoadObjectDetector:
    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.35, imgsz: int = 640):
        # Weights are downloaded once on first use, then cached locally.
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, frame) -> List[Detection]:
        result = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            classes=list(ROAD_CLASSES),
            device="cpu",
            verbose=False,
        )[0]
        detections: List[Detection] = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(ROAD_CLASSES[cls_id], float(box.conf[0]), x1, y1, x2, y2)
            )
        return detections
