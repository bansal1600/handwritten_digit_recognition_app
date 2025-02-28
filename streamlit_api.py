from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import uvicorn
import os

app = FastAPI()

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "digit_recognition_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)

class ImageData(BaseModel):
    image: str  # Base64-encoded image string

def preprocess_image(image_data: str):
    """Convert base64 image data to a preprocessed 28x28 grayscale image."""
    img = Image.open(BytesIO(base64.b64decode(image_data.split(",")[1]))).convert("L")
    img = img.resize((28, 28))
    img = np.array(img)

    # Invert and normalize
    img = 255 - img  # Convert black-on-white to white-on-black
    img = img.astype(np.float32) / 255.0  # Normalize

    # Reshape for model input
    img = img.reshape(1, 28, 28, 1)
    return img

@app.post("/predict")
async def predict_digit(data: ImageData):
    """Predict the digit from the drawn image."""
    try:
        img = preprocess_image(data.image)
        prediction = model.predict(img)
        predicted_digit = int(np.argmax(prediction))
        return {"digit": predicted_digit}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
