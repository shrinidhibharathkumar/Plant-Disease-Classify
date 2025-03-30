import cv2
import numpy as np

def annotate_image(image: np.ndarray, label: str, conf: float) -> np.ndarray:
    """Annotate the image with label and confidence."""
    text = f"{label} ({conf*100:.2f}%)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)
    padding = 10

    img_h, img_w = image.shape[:2]
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    box_w = text_w + 2 * padding
    box_h = text_h + baseline + 2 * padding

    while (box_w > img_w or box_h > img_h) and font_scale > 0.3:
        font_scale -= 0.1
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        box_w = text_w + 2 * padding
        box_h = text_h + baseline + 2 * padding

    x1, y1 = 0, 0
    x2, y2 = x1 + box_w, y1 + box_h
    cv2.rectangle(image, (x1, y1), (x2, y2), bg_color, -1)
    text_x = x1 + padding
    text_y = y1 + padding + text_h
    cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, thickness)
    return image
