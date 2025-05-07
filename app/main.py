from fastapi import FastAPI, UploadFile, File
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import torch
import io

app = FastAPI()

MODEL_NAME = "Falah/vit-base-breast-cancer"
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

@app.get("/")
async def root():
    return {"message": "API de clasificación de cáncer de mama (local)"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = probs.argmax().item()
        confidence = probs[0][predicted_class].item()

    return {
        "predicted_class": model.config.id2label.get(predicted_class, str(predicted_class)),
        "confidence": round(confidence, 4)
    }


