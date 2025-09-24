import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Configuration MLflow pour GitHub Actions
# -----------------------------
# Stocker les artefacts dans un dossier relatif au repo
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"

# -----------------------------
# Charger le dataset Iris
# -----------------------------
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Entraîner le modèle
# -----------------------------
model = RandomForestClassifier(n_estimators=400, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Évaluer le modèle
# -----------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc}")

# -----------------------------
# Logger dans MLflow
# -----------------------------
mlflow.set_experiment("iris-classification")
with mlflow.start_run():
    mlflow.log_param("n_estimators", 400)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "iris_model")
