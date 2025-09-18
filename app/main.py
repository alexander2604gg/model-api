import os
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from typing import List
from concurrent.futures import ThreadPoolExecutor

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

classifier = pipeline(
    "text-classification",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR
)

app = FastAPI(title="Depression Classifier API")

class TextRequest(BaseModel):
    redditId: str
    text: str

class TextBatchRequest(BaseModel):
    texts: List[TextRequest]

class PredictionResponse(BaseModel):
    redditId: str
    text: str
    prediction: dict

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

@app.get("/")
def root():
    return {"message": "Depression Classification API is running"}

BATCH_SIZE = 5

def classify_batch(texts: List[str]) -> List[dict]:
    return classifier(texts)

@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(request: TextBatchRequest):
    print(f"Textos recibidos: {len(request.texts)}")
    responses = []

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as executor:
        for i in range(0, len(request.texts), BATCH_SIZE):
            batch = request.texts[i:i + BATCH_SIZE]
            texts = [item.text for item in batch]
            results = await loop.run_in_executor(executor, classify_batch, texts)

            for item, result in zip(batch, results):
                responses.append({
                    "redditId": item.redditId,
                    "text": item.text,
                    "prediction": result
                })

    return {"predictions": responses}