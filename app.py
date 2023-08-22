from fastapi import FastAPI, HTTPException
import joblib
import numpy as np

app = FastAPI()

# Load the pre-trained model during startup
@app.on_event("startup")
def load_model():
    global model
    model = joblib.load("model.pkl")

# Prediction route
@app.post("/predict/")
def predict(features: dict):
    # Validate input features
    required_features = ["LotArea", "MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GrLivArea", "GarageArea", "WoodDeckSF", "OpenPorchSF"]
    for feature in required_features:
        if feature not in features:
            raise HTTPException(status_code=400, detail=f"'{feature}' is required")

    # Prepare input data for prediction
    input_data = np.array([
        features["LotArea"],
        features["MasVnrArea"],
        features["BsmtUnfSF"],
        features["TotalBsmtSF"],
        features["1stFlrSF"],
        features["2ndFlrSF"],
        features["GrLivArea"],
        features["GarageArea"],
        features["WoodDeckSF"],
        features["OpenPorchSF"]
    ]).reshape(1, -1)  # Convert to a 2D array

    # Make prediction using the loaded model
    predicted_price = model.predict(input_data)[0]

    return {"predicted_price": predicted_price}
