from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow.sklearn

# Charger données
X, y = load_iris(return_X_y=True)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Charger dernier modèle loggé
model = mlflow.sklearn.load_model("runs:/7a327c59dbc74331a0f9234cbe88dc07/iris_model")

# Tester
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy du modèle en Registry : {acc}")
