import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline


# Construye la ruta absoluta al modelo
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

# Inicializa el pipeline usando la ruta absoluta
classifier = pipeline(
    "text-classification",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR
)
app = FastAPI(title="Depression Classifier API")

class TextRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Depression Classification API is running"}

@app.post("/predict")
def predict(request: TextRequest):
    result = classifier(request.text)
    return {"text": request.text, "prediction": result[0]}
