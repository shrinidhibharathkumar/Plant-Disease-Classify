from ultralytics import YOLO
import numpy as np

def load_model():
    """Load the YOLO model."""
    return YOLO("models/yolo11n/yolo11n-cls.onnx")

def predict(model, img: np.ndarray):
    """Predict the class label and confidence score."""
    results = model.predict(img, conf=0.75)
    pred_idx = results[0].probs.top1
    pred_label = model.names[pred_idx]
    pred_conf = results[0].probs.top1conf
    return pred_label, pred_conf
