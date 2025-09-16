from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Carga del modelo y tokenizer
classifier = pipeline(
    "text-classification",
    model="app/model/depression_model",
    tokenizer="app/model/depression_model"
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
