from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

@app.on_event("startup")
def load_model():
    global model
    model = joblib.load("model.pkl")

@app.post("/predict/")
def predict(data: dict):
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": prediction.tolist()}
