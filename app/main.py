from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from app.model.yolo_model import load_model, predict
from app.utils.image_io import read_image_from_upload, read_image_from_url
from app.utils.annotate import annotate_image
from io import BytesIO
import cv2

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

    # Encode the annotated image as JPEG
    _, buffer = cv2.imencode('.jpg', annotated_img)
    return StreamingResponse(BytesIO(buffer.tobytes()), media_type="image/jpeg")
