## AI Image Classifier

Interactive Streamlit app that uses a pre-trained MobileNetV2 model (ImageNet weights) to classify uploaded images and show the top 3 predictions with confidence scores.

### Features
- Upload JPG/PNG images from the browser.
- Runs MobileNetV2 locally with TensorFlow.
- Shows top-3 decoded labels with confidence.
- Simple, single-file app (`main.py`).

### Prerequisites
- Python 3.13+
- git (if cloning)
- Recommended: virtual environment (`python -m venv .venv && .\\.venv\\Scripts\\activate` on Windows, `source .venv/bin/activate` on macOS/Linux)

### Setup
Using [uv](https://docs.astral.sh/uv/) (recommended):
1) Install uv: `pip install uv`
2) Install deps: `uv sync`

Using pip:
1) `pip install streamlit tensorflow opencv-python pillow`
   - TensorFlow wheels can be large; allow time for the download.

### Run the app
- With uv: `uv run streamlit run main.py`
- With pip: `streamlit run main.py`

Then open the local URL Streamlit prints (usually http://localhost:8501), upload an image, and click **Classify Image** to see predictions.

### Notes
- Model weights download on first run; ensure internet access.
- Cached model loading is handled via `st.cache_resource` to avoid reloading between interactions.
