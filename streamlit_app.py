import streamlit as st
import requests
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# FastAPI server URL
API_URL = "http://127.0.0.1:8000/predict"

st.title("Digit Recognition App 🎨🔢")

# Initialize session state variables
if "predicted_digit" not in st.session_state:
    st.session_state["predicted_digit"] = None
if "canvas_key" not in st.session_state:
    st.session_state["canvas_key"] = "canvas_1"

# Function to convert image to Base64 for API
def image_to_base64(image_array):
    img = Image.fromarray(image_array.astype("uint8"))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()

# Display drawing canvas (white background, larger size)
canvas_result = st_canvas(
    stroke_width=10,
    height=350,
    width=350,
    key=st.session_state["canvas_key"],
    background_color="#FFFFFF",
    update_streamlit=True
)

# Predict button logic
if st.button("Predict"):
    if canvas_result.image_data is not None:
        image_data = np.array(canvas_result.image_data[:, :, :3])  # Remove alpha channel

        # Check if there is a drawing (not just a blank canvas)
        if np.any(image_data != 255):  
            base64_image = image_to_base64(image_data)
            response = requests.post(API_URL, json={"image": base64_image})

            if response.status_code == 200:
                st.session_state["predicted_digit"] = response.json().get("digit")
            else:
                st.session_state["predicted_digit"] = "Error in prediction!"

# Display the prediction only if "Predict" was clicked
if st.session_state["predicted_digit"] is not None:
    st.subheader(f"Predicted Digit: **{st.session_state['predicted_digit']}**")

# Clear button logic
if st.button("Clear"):
    st.session_state["canvas_key"] = f"canvas_{np.random.randint(1000)}"  # Reset canvas
    st.session_state["predicted_digit"] = None  # Remove prediction text
    st.rerun()