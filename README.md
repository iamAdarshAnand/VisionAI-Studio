# 👁️ VisionAI Studio

An interactive **computer vision workbench** built with Streamlit, OpenCV and Pillow. Upload an image and experiment with classic image processing and optional object detection.

## Features
- Grayscale, Gaussian blur, Canny edges, adaptive thresholding and contours
- Adjustable processing controls and before/after comparison
- Download processed images as PNG
- Optional object detection with a locally supplied ONNX model (not bundled)
- No API keys; images are processed in the running Python environment
- Unit tests for image-processing functions

## Quick start
```bash
git clone https://github.com/iamAdarshAnand/VisionAI-Studio.git
cd VisionAI-Studio
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Modes
- **Grayscale**: luminance conversion
- **Gaussian blur**: noise reduction with adjustable odd-sized kernel
- **Canny edges**: edge map using low/high thresholds
- **Adaptive threshold**: local binary segmentation
- **Contours**: draw detected external contours

### Optional object detection
The `detect_objects` function in `vision.py` supports OpenCV DNN inference for **compatible YOLOv5-style ONNX outputs**. Supply a trusted model file yourself; model weights are not bundled. Model architectures and output layouts vary, so this is an optional developer API rather than a guaranteed plug-and-play UI feature.

## Tests
```bash
pytest -q
```

## Privacy
Do not upload private photos to public hosting. No images are sent to third-party inference APIs by the application.

Created by Adarsh Anand.

## Project discovery

**Computer Vision Image Processing Studio** — Streamlit OpenCV image processing app with Canny edge detection, grayscale filters, Gaussian blur, contours, adaptive thresholding and optional ONNX object detection.

**Related developer searches:** computer vision, image processing, OpenCV, Streamlit, Python, edge detection, ONNX.

**Source:** [VisionAI-Studio](https://github.com/iamAdarshAnand/VisionAI-Studio) · **Author:** [Adarsh Anand](https://github.com/iamAdarshAnand)
