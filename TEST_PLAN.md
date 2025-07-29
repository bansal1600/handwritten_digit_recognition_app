# Test Plan for Handwritten Digit Recognition Application

This document outlines the strategy for testing the FastAPI backend, the Streamlit frontend, and the overall integration of the handwritten digit recognition system.

## 1. Setup

1. **Install dependencies**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Launch the application**
   ```bash
   streamlit run streamlit_app.py
   ```
   The Streamlit app automatically starts the FastAPI server in a subprocess.

## 2. Testing Framework

- Use **pytest** for unit and integration tests.
- Use **requests** to call the FastAPI endpoint during tests.
- Use **streamlit testing utilities** (e.g., `pytest-streamlit`) for basic UI checks.

## 3. Unit Tests

1. **FastAPI endpoint**
   - Test `/predict` with a valid base64 encoded digit image.
   - Test `/predict` with malformed or invalid data (expect an error message).

2. **Preprocessing function**
   - Verify `preprocess_image` converts images to 28x28 grayscale and normalizes values to `[0, 1]`.

## 4. Integration Tests

1. **Backend and Model**
   - Start FastAPI using `uvicorn` in a background thread and send sample requests.
   - Check that a correct digit is returned for known inputs.

2. **Streamlit Frontend**
   - Use a headless browser (e.g., `pytest-streamlit` or `selenium`) to ensure the canvas renders and that clicking the Predict button triggers a call to the API.
   - Verify that the predicted digit is displayed on the page.

## 5. Manual Testing

- Run the Streamlit app locally and draw digits on the canvas to confirm real-time predictions.
- Validate edge cases such as partially erased digits and rapid predictions.

## 6. Continuous Integration (Optional)

- Add a GitHub Actions workflow to run tests on each push.
- Cache the model file to avoid re-downloading or retraining during CI runs.

