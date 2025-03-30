import cv2
import numpy as np
import requests
from fastapi import UploadFile

def read_image_from_upload(uploaded_file: UploadFile) -> np.ndarray:
    """Read an image from an uploaded file."""
    file_bytes = uploaded_file.file.read()
    file_array = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    return cv2.imdecode(file_array, cv2.IMREAD_COLOR)

def read_image_from_url(url: str) -> np.ndarray:
    """Read an image from a URL."""
    response = requests.get(url)
    file_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    return cv2.imdecode(file_array, cv2.IMREAD_COLOR)
