import os
import joblib
import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

logging.basicConfig(level=logging.INFO)

###########################################################
# SageMaker Directories
###########################################################

TRAIN_DIR = "/opt/ml/input/data/train"
MODEL_DIR = "/opt/ml/model"

###########################################################
# Read Features
###########################################################

train_file = os.path.join(TRAIN_DIR, "features.csv")

logging.info(f"Reading feature file : {train_file}")

df = pd.read_csv(train_file)

logging.info(f"Dataset Shape : {df.shape}")

###########################################################
# Target Column
###########################################################

TARGET_COLUMN = "CompliancePercentage"

if TARGET_COLUMN not in df.columns:
    raise Exception(f"{TARGET_COLUMN} column not found.")

###########################################################
# Remove non-feature columns
###########################################################

DROP_COLUMNS = [
    TARGET_COLUMN,
    "AccountId",
    "AccountName"
]

existing_drop_columns = [
    c for c in DROP_COLUMNS if c in df.columns
]

X = df.drop(columns=existing_drop_columns)

y = df[TARGET_COLUMN]

###########################################################
# Train/Test Split
###########################################################

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

###########################################################
# Train Model
###########################################################

logging.info("Training RandomForestRegressor...")

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

###########################################################
# Predictions
###########################################################

predictions = model.predict(X_test)

###########################################################
# Metrics
###########################################################

mae = mean_absolute_error(y_test, predictions)

rmse = mean_squared_error(
    y_test,
    predictions,
    squared=False
)

r2 = r2_score(
    y_test,
    predictions
)

logging.info("--------------------------------")
logging.info(f"MAE  : {mae:.4f}")
logging.info(f"RMSE : {rmse:.4f}")
logging.info(f"R2   : {r2:.4f}")
logging.info("--------------------------------")

###########################################################
# Save Model
###########################################################

os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(
    MODEL_DIR,
    "model.joblib"
)

joblib.dump(model, model_path)

logging.info(f"Model saved to {model_path}")

logging.info("Training completed successfully.")