from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import io
import numpy as np
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

def load_model():
    model = joblib.load("trained_model.pkl")
    
model = load_model()

def preprocess_image(image: Image.Image):
    image = image.resize((128, 128))
    image_np = np.array(image).flatten()
    return image_np

def predict(image_array):
    prediction = model.predict([image_array])
    return prediction[0]

@app.post('/predict')
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("L")
    image_array = preprocess_image(image)
    prediction = predict(image_array).tolist()
    return {'prediction': prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)