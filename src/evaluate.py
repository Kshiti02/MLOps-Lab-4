import pandas as pd
import pickle
import yaml
import json
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os
import sys

# Ensure experiments directory exists
os.makedirs("experiments", exist_ok=True)

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    logger.info("Loading parameters...")
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
except FileNotFoundError:
    logger.error("Error: params.yaml not found")
    sys.exit(1)

logger.info("Loading test data...")
test_data = pd.read_csv(params["data"]["test"])
X_test = test_data.drop(columns=[params["data"]["target_col"]])
y_test = test_data[params["data"]["target_col"]]

if not os.path.exists(params["data"]["model"]):
    logger.error("Error: Model file not found")
    sys.exit(1)

logger.info("Loading model...")
with open(params["data"]["model"], "rb") as f:
    model = pickle.load(f)

logger.info("Predicting...")
y_pred = model.predict(X_test)

logger.info("Calculating metrics...")
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
metrics = {"accuracy": accuracy, "f1": f1}

logger.info("Saving metrics...")
with open("metrics.json", "w") as f:
    json.dump(metrics, f)
with open("experiments/results.json", "w") as f:
    json.dump(metrics, f)

logger.info("Saving confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(cm).to_csv("experiments/confusion_matrix.csv", index=False)
logger.info("Evaluation complete.")