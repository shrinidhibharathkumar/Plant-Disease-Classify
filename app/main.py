from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.model.yolo_model import load_model, predict
from app.utils.image_io import read_image_from_upload, read_image_from_url
from app.utils.annotate import annotate_image
from io import BytesIO
import cv2
import base64

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the YOLO model
model = load_model()

@app.post("/process")
async def process_image(image: UploadFile = File(None), imageUrl: str = Form(None)):
    if image:
        img = read_image_from_upload(image)
    elif imageUrl:
        img = read_image_from_url(imageUrl)
    else:
        return {"error": "No image provided"}

    label, conf = predict(model, img)
    annotated_img = annotate_image(img, label, conf)
    confidence_float = float(conf)

    # Encode image
    _, buffer = cv2.imencode('.jpg', annotated_img)
    image_bytes = BytesIO(buffer.tobytes())
    base64_image = base64.b64encode(image_bytes.read()).decode("utf-8")

    return {
        "image": base64_image,
        "label": label,
        "confidence": confidence_float*100
    }
