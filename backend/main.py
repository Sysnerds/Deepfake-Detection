
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from fastapi.middleware.cors import CORSMiddleware

from src.model import BasicCNN
from src.config import MODEL_DIR, IMAGE_SIZE


app = FastAPI(
    title="AI-Generated Image Detector",
    description="Classifies facial images as real or AI-generated.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Use the GPU when available, otherwise use the CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build the same CNN architecture used during training.
model = BasicCNN(num_classes=2)

CHECKPOINT_PATH = MODEL_DIR / "best_basic_cnn.pth"

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found: {CHECKPOINT_PATH}"
    )

# Load the trained weights.
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False,
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

# Match validation/test preprocessing. Do not apply random augmentation.
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


@app.get("/")
def home():
    return {"message": "AI-Generated Image Detector API is running!"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(DEVICE),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(
            status_code=415,
            detail="Please upload a JPEG, PNG, or WebP image.",
        )

    try:
        contents = await file.read()

        # Reject very large uploads before decoding.
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image must be 10 MB or smaller.",
            )

        import io
        image = Image.open(io.BytesIO(contents)).convert("RGB")

    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    # Dataset mapping: 0 = fake, 1 = real.
    predicted_class = int(probabilities.argmax().item())
    label = "REAL" if predicted_class == 1 else "AI_GENERATED"
    confidence = float(probabilities[predicted_class].item())

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "probabilities": {
            "AI_GENERATED": round(float(probabilities[0].item()), 4),
            "REAL": round(float(probabilities[1].item()), 4),
        },
        "model": "BasicCNN",
    }
