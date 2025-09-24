import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Charger dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner modèle
model = RandomForestClassifier(n_estimators=400, random_state=42)
model.fit(X_train, y_train)

# Évaluer
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Logger dans MLflow
mlflow.set_experiment("iris-classification 2")
with mlflow.start_run():
    mlflow.log_param("n_estimators", 400)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "iris_model")
