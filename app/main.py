import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from typing import List

# Construye la ruta absoluta al modelo
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

# Inicializa el pipeline usando la ruta absoluta
classifier = pipeline(
    "text-classification",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR
)

app = FastAPI(title="Depression Classifier API")

# ---- MODELOS DE REQUEST ----
class TextRequest(BaseModel):
    redditId: str
    text: str

class TextBatchRequest(BaseModel):
    texts: List[TextRequest]

# ---- MODELOS DE RESPONSE ----
class PredictionResponse(BaseModel):
    redditId: str
    text: str
    prediction: dict  # contiene {"label": "...", "score": ...}

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

@app.get("/")
def root():
    return {"message": "Depression Classification API is running"}

@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(request: TextBatchRequest):
    print(f"Textos recibidos: {len(request.texts)}")

    texts = [item.text for item in request.texts]  # extraemos solo el texto
    results = classifier(texts)

    response = [
        {
            "redditId": item.redditId,
            "text": item.text,
            "prediction": result
        }
        for item, result in zip(request.texts, results)
    ]
    return {"predictions": response}
