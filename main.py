from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("../models/xgboost_model.pkl")


@app.get("/")
def home():
    return {"message": "Sales Forecasting API Running"}


@app.get("/predict")
def predict():

    sample_data = pd.DataFrame({
        'month': [1],
        'day_of_week': [2],
        'quarter': [1],
        'lag_1': [500000000],
        'lag_7': [480000000],
        'lag_30': [450000000],
        'rolling_mean_7': [470000000],
        'rolling_std_7': [15000000]
    })

    prediction = model.predict(sample_data)

    return {
        "forecasted_sales": float(prediction[0])
    }