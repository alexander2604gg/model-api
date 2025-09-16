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
async def predict(request: TextRequest):
    print(f"Texto recibido: {request.text}")  # Debug
    result = classifier(request.text)
    print(f"Resultado: {result}")  # Debug
    return {"text": request.text, "prediction": result[0]}
