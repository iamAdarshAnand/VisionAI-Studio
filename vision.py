"""Image-processing primitives and optional ONNX inference helper."""
from __future__ import annotations
import cv2
import numpy as np

MODES = ("Original", "Grayscale", "Gaussian blur", "Canny edges", "Adaptive threshold", "Contours")

def ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Expected a numpy image")
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("Expected a non-empty H x W x 3 RGB image")
    return np.ascontiguousarray(image.astype(np.uint8))

def process_image(image: np.ndarray, mode: str, blur_kernel: int = 5, canny_low: int = 60, canny_high: int = 160) -> np.ndarray:
    rgb = ensure_rgb(image)
    if blur_kernel < 1 or blur_kernel % 2 == 0:
        raise ValueError("Blur kernel must be a positive odd number")
    if not (0 <= canny_low <= canny_high <= 255):
        raise ValueError("Canny thresholds must satisfy 0 <= low <= high <= 255")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if mode == "Original":
        return rgb.copy()
    if mode == "Grayscale":
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    if mode == "Gaussian blur":
        return cv2.GaussianBlur(rgb, (blur_kernel, blur_kernel), 0)
    if mode == "Canny edges":
        edges = cv2.Canny(gray, canny_low, canny_high)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    if mode == "Adaptive threshold":
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    if mode == "Contours":
        edges = cv2.Canny(gray, canny_low, canny_high)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = rgb.copy()
        cv2.drawContours(result, contours, -1, (0, 255, 120), 2)
        return result
    raise ValueError(f"Unknown processing mode: {mode}")

def detect_objects(image: np.ndarray, onnx_path: str, confidence: float = 0.4, nms_threshold: float = 0.45) -> list[tuple[int, float, tuple[int, int, int, int]]]:
    """Optional YOLOv5-format ONNX detection; returns (class_id, confidence, xywh).

    Expects output shaped [1, N, 5 + num_classes] with normalized/model-space
    xywh, objectness, and class probabilities. Other model exports need adapters.
    """
    rgb = ensure_rgb(image)
    if not 0 < confidence <= 1 or not 0 <= nms_threshold <= 1:
        raise ValueError("Invalid detection thresholds")
    height, width = rgb.shape[:2]
    net = cv2.dnn.readNetFromONNX(onnx_path)
    blob = cv2.dnn.blobFromImage(rgb, scalefactor=1 / 255.0, size=(640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    output = np.asarray(net.forward())
    predictions = np.squeeze(output, axis=0) if output.ndim == 3 and output.shape[0] == 1 else output
    if predictions.ndim != 2 or predictions.shape[1] < 6:
        raise ValueError("Unsupported ONNX output format")
    boxes, scores, classes = [], [], []
    for row in predictions:
        objectness = float(row[4])
        class_id = int(np.argmax(row[5:]))
        score = objectness * float(row[5 + class_id])
        if score < confidence:
            continue
        cx, cy, w, h = row[:4]
        x = int((cx - w / 2) * width / 640)
        y = int((cy - h / 2) * height / 640)
        boxes.append([x, y, int(w * width / 640), int(h * height / 640)])
        scores.append(score)
        classes.append(class_id)
    if not boxes:
        return []
    keep = cv2.dnn.NMSBoxes(boxes, scores, confidence, nms_threshold)
    return [(classes[int(i)], scores[int(i)], tuple(boxes[int(i)])) for i in np.asarray(keep).flatten()]
