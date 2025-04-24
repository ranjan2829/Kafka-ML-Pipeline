from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open("iris_model.pkl", "wb") as f:
    pickle.dump(model, f)
