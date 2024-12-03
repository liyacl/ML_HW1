import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import joblib
from pydantic import BaseModel, ValidationError, Field
import uvicorn
from fastapi.responses import StreamingResponse
import io
import re

app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]

try:
    model = joblib.load("linear_model_last.joblib")
    feature_scaler = joblib.load("feature_scaler.joblib")
except FileNotFoundError:
    print("Файлы модели или стандартизаторов не найдены :(")
    exit(1)

def prepros_nan(x):
    x = str(x).split(' ')[0] if x is not np.nan else np.nan
    try:
        x = float(x)
    except ValueError:
        x = np.nan
    return x


def preprocess_data(data):

    # 1. предобработка данных
    for col in data.columns:
        data[col] = data[col].apply(prepros_nan)
    data = data.drop_duplicates(keep='first')
    data = data.reset_index(drop=True)
    
    for col in ['mileage', 'engine', 'max_power']:
        data[col] = data[col].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
    return data


def find_float(elem):
    return float(re.findall(r'(\d+\.?\d*)', elem)[0])

@app.post("/predict_item") 
def predict_item(item: Item): 
    item = item.dict()
    data = {
        'year': [float(item['year'])],
        'km_driven': [float(item['km_driven'])],
        'mileage': [find_float(item['mileage'])],
        'engine': [find_float(item['engine'])],
        'max_power': [find_float(item['max_power'])],
        'seats': [float(item['seats'])]
    }
    data = pd.DataFrame(data)
    formatting_data = preprocess_data(data) 
    scaled_features = feature_scaler.transform(formatting_data)

    prediction = model.predict(scaled_features)
    return prediction.tolist()

@app.post("/predict_items", response_class=StreamingResponse) 
async def predict_items(file: UploadFile = File(...)): 
    content = await file.read() 
    df_1 = pd.read_csv(io.BytesIO(content))

    numeric_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
    df = df_1.copy()[numeric_cols]
 
    formatting = preprocess_data(df) 
    scaled_features = feature_scaler.transform(formatting)
    predictions = model.predict(scaled_features)
    
    df['predictions_price'] = predictions
 
    output_stream = io.StringIO() 
    df.to_csv(output_stream, index=False) 
    output_stream.seek(0) 
 
    response = StreamingResponse(iter([output_stream.getvalue()]), media_type="text/csv") 
    response.headers["Content-Disposition"] = "attachment; filename=predictions_output.csv" 
    return response
    
if __name__ == "__main__":
    uvicorn.run('main:app', port=8000, reload=True)
