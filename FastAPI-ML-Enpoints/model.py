from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
iris=load_iris()
x,y=    iris.data,iris.target
model=RandomForestClassifier()
model.fit(x,y)
joblib.dump(model,'model.pkl')