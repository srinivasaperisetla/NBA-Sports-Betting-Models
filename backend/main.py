from fastapi import FastAPI, HTTPException, Query
import pandas as pd
from testModel_schema import TestModelPredictionInput, Stat
from constants import TARGET_COLUMNS, FEATURES_TO_DUMMY
from contextlib import asynccontextmanager
import tensorflow as tf
import xgboost as xgb
import pickle
import uvicorn
import json
import os
import joblib

ALLSTAR_PATH_KERAS = "./models/allstar.keras"
CHAMPION_MODELS_DIR = "./models/champion_models"

ALLSTAR_SCALER = "./scalers/allstar_scaler.pkl"
ALLSTAR_FEATURES = "./features/allstar_features.pkl"
CATEGORY_MAPPINGS = "./category_mappings/category_mappings.json"

CHAMPION_FEATURES = "./features/champion_features.json"

testModel = None
champion_models = {}
testModel_scaler = None
testModel_columns = None
category_mappings = None
champion_feature_names = None

@asynccontextmanager
async def lifespan(app: FastAPI):
  global allstar_model, allstar_scaler, allstar_features, category_mappings, champion_feature_names, champion_models
  print("🔄 Loading model and assets...")

  # testModel = tf.keras.layers.TFSMLayer(TESTMODEL_PATH, call_endpoint='serving_default')
  allstar_model = tf.keras.models.load_model(ALLSTAR_PATH_KERAS)

  with open(ALLSTAR_SCALER, 'rb') as f:
    allstar_scaler = pickle.load(f)

  with open(ALLSTAR_FEATURES, 'rb') as f:
    allstar_features = pickle.load(f)

  with open(CATEGORY_MAPPINGS, 'r') as f:
    category_mappings = json.load(f)
  
  with open(CHAMPION_FEATURES,"r") as f:
    champion_feature_names = json.load(f)
  
  files = os.listdir(CHAMPION_MODELS_DIR)

  for filename in files:
    stat_name = filename.replace("_calibrated.pkl", "")
    model_path = os.path.join(CHAMPION_MODELS_DIR, filename)

    champion_models[stat_name] = joblib.load(model_path)
    
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

@app.post("/predict_allstar")
def predict(input_data: TestModelPredictionInput):
  try:
    df = pd.DataFrame([input_data.model_dump(by_alias=True)])

    df["PLAYER_NAME_ID"] = category_mappings["PLAYER_NAME"].get(df["PLAYER_NAME"].iloc[0], -1)
    df["POSITION_ID"] = category_mappings["POSITION"].get(df["POSITION"].iloc[0], -1)
    df["TEAM_ID"] = category_mappings["TEAM"].get(df["TEAM"].iloc[0], -1)
    df["MATCHUP_ID"] = category_mappings["MATCHUP"].get(df["MATCHUP"].iloc[0], -1)

    df = df.drop(columns=["PLAYER_NAME", "TEAM", "MATCHUP", "POSITION"])

    df = df[allstar_features]

    embedded_cols = ["PLAYER_NAME_ID", "TEAM_ID", "MATCHUP_ID", "POSITION_ID"]
    numerical_cols = [c for c in df.columns if c not in embedded_cols]
  
    numeric_scaled = allstar_scaler.transform(df[numerical_cols])

    model_input = {
      "PLAYER_NAME_ID": df["PLAYER_NAME_ID"].values,
      "TEAM_ID": df["TEAM_ID"].values,
      "MATCHUP_ID": df["MATCHUP_ID"].values,
      "POSITION_ID": df["POSITION_ID"].values,
      "NUMERIC": numeric_scaled
    }

    outputs = allstar_model(model_input, training=False)

    results = []

    for stat, out in zip(TARGET_COLUMNS, outputs):
      prob = float(out.numpy()[0][0])
      prediction = int(prob > 0.5)
      confidence_prob = prob if prediction == 1 else (1 - prob)
      parlay = getattr(input_data, f"PL_{stat}")

      results.append({
        "stat": stat,
        "parlay": parlay,
        "prediction": prediction,
        "probability": round(prob, 3),
        "confidence": f"{round(confidence_prob * 100, 1)}%"
      })

    return {"predictions": results}

  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
  
@app.post('/predict_champion')
def predict_champion(input_data: TestModelPredictionInput, stat: Stat = Query(...)):
  try:
    stat_name = stat.value
    xgb_model = champion_models[stat_name]
    parlay = getattr(input_data, f"PL_{stat_name}")

    df = pd.DataFrame([input_data.model_dump(by_alias=True)])

    cols_to_drop = []
    for col in TARGET_COLUMNS:
      if col != stat_name:
        cols_to_drop.append(f'PL_{col}')
        cols_to_drop.append(f'OVER_PL_RATE_{col}_LAST10')
        cols_to_drop.append(f'OVER_PL_RATE_{col}_LAST5')
        cols_to_drop.append(f'{col}_Z_LINE')
        cols_to_drop.append(f'{col}_Z_RECENT')
        cols_to_drop.append(f'{col}_LINE_DIFF_X_MIN')
        cols_to_drop.append(f'LINE_EDGE_{col}')
        cols_to_drop.append(f'LINE_AMBIGUITY_{col}')
      
    df = df.drop(columns = cols_to_drop)

    expected_cols = champion_feature_names[stat_name]
    df = df.reindex(columns=expected_cols, fill_value=0)

    prob = float(xgb_model.predict_proba(df)[0, 1])

    prediction = int(prob > 0.5)

    return {
      "stat": stat_name,
      "parlay": parlay,
      "prediction": prediction,
      "probability": round(prob, 2),
    }

  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


# for local use only

if __name__ == "__main__":
  uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)