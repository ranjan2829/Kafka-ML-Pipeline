# train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Simulate a dataset (you can replace with real one later)
data = {
    'feature1': [5.1, 4.9, 6.2, 7.0, 5.9, 6.7, 6.0, 5.4],
    'feature2': [3.5, 3.0, 2.9, 3.2, 3.0, 3.1, 2.2, 3.4],
    'feature3': [1.4, 1.4, 4.3, 4.7, 5.1, 4.4, 5.0, 1.7],
    'label':    [0, 0, 1, 1, 2, 2, 2, 0]
}
df = pd.DataFrame(data)

# Split the data
X = df[['feature1', 'feature2', 'feature3']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save the model
os.makedirs("model", exist_ok=True)
joblib.dump(clf, "model/trained_model.pkl")
print("Model saved to model/trained_model.pkl")
