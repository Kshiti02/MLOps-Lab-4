
import pandas as pd
import pickle
import yaml
import json
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os

# Ensure experiments directory exists
os.makedirs("experiments", exist_ok=True)

print("Loading parameters...")
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

    print("Loading test data...")
    test_data = pd.read_csv(params["data"]["test"])
    X_test = test_data.drop(columns=[params["data"]["target_col"]])
    y_test = test_data[params["data"]["target_col"]]

    print("Loading model...")
    with open(params["data"]["model"], "rb") as f:
         model = pickle.load(f)

    print("Predicting...")
    y_pred = model.predict(X_test)

    print("Calculating metrics...")
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    metrics = {"accuracy": accuracy, "f1": f1}
     
    print("Saving metrics...")
    with open("metrics.json", "w") as f:
         json.dump(metrics, f)
    with open("experiments/results.json", "w") as f:
         json.dump(metrics, f)

    print("Saving confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(cm).to_csv("experiments/confusion_matrix.csv", index=False)
print("Evaluation complete.")