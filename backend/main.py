from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from testModel_schema import TestModelPredictionInput
from constants import TARGET_COLUMNS, FEATURES_TO_DUMMY
from contextlib import asynccontextmanager
import tensorflow as tf
import pickle
import uvicorn

TESTMODEL_PATH = "./models/testModel"
TESTMODEL_PATH_KERAS = "./models/testModel.keras"
TESTMODEL_SCALER = "./scalers/testModel_scaler.pkl"
TESTMODEL_COLUMNS = "./columns/testModel_columns.pkl"

testModel = None
testModel_scaler = None
testModel_columns = None

@asynccontextmanager
async def lifespan(app: FastAPI):
  global testModel, testModel_scaler, testModel_columns
  print("🔄 Loading model and assets...")

  # testModel = tf.keras.layers.TFSMLayer(TESTMODEL_PATH, call_endpoint='serving_default')
  testModel = tf.keras.models.load_model(TESTMODEL_PATH_KERAS)

  with open(TESTMODEL_SCALER, 'rb') as f:
    testModel_scaler = pickle.load(f)

  with open(TESTMODEL_COLUMNS, 'rb') as f:
    testModel_columns = pickle.load(f)

  print("✅ Assets loaded.")
  yield
  print("🛑 Shutting down app...")

app = FastAPI(
  title="NBA Player Prop Prediction API",
  description="Predicts various NBA player statistics for a given game using a TensorFlow ANN model.",
  lifespan=lifespan
)

@app.get("/")
def root():
  return {"status": "API is live"}

@app.post("/predict")
def predict(input_data: TestModelPredictionInput):
  try:
    df = pd.DataFrame([input_data.model_dump(by_alias=True)])
    df = pd.get_dummies(df, columns=FEATURES_TO_DUMMY)

    for col in testModel_columns:
      if col not in df.columns:
        df[col] = 0
    df = df[testModel_columns]

    X_scaled = testModel_scaler.transform(df)
    prediction = testModel.predict(X_scaled)

    output = {
      target: [
        int(prob > 0.5),
        f"{round(prob * 100, 1)}%"
      ]
      for target, prob in zip(TARGET_COLUMNS, prediction[0])
    }

    return {"prediction": output}

  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

# for local use only

# if __name__ == "__main__":
#   uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)