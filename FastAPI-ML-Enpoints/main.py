from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model=joblib.load('model.pkl')
app = FastAPI()
class requestModel(BaseModel):
    sepal_length:float
    sepal_width:float
    petal_length:float
    petal_width:float
@app.get('/')
def read_root():
    return {'message':'Hello World'}

@app.post('/predict')
def predict(data:requestModel):
    data =np.array([[data.sepal_length,data.sepal_width,data.petal_length,data.petal_width]])
    prediction = model.predict(data)
    species=['setosa','versicolor','virginica']
    return {'prediction':species[prediction[0]]}

   
