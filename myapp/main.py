import numpy as np
from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
from myapp.my_model import Model
import uvicorn
import logging
from pydantic import BaseModel
import pandas as pd
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Form(BaseModel):
    image: list
class Prediction(BaseModel):
    prediction: int
# load model
model = Model()

# app
app = FastAPI(title='Symbol detection', docs_url='/docs')

@app.get('/status')
def status():
    return 'i am OK'
# api
@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame(np.array(dict(form)['image']).reshape(1,-1))
    pred = model.predict(df)
    print(pred[0])
    return {'prediction': pred[0]}

# static files
#app.mount('/', StaticFiles(directory='static', html=True), name='static')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("myapp.main:app", host="0.0.0.0", port=8000, reload=True)