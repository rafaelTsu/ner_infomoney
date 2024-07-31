from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
from transformers import pipeline

app = FastAPI()

model_path = 'script\output\checkpoint-6'
classifier = pipeline('token-classification', model=model_path, tokenizer=model_path)

class PredictionRequest(BaseModel):
    text: List[str]

class PredictionResponse(BaseModel):
    predictions: List[dict]

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/setup")
async def setup(request: Request):
    data = await request.json()
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    texts = request.text
    predictions = classifier(texts)
    return {"predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
