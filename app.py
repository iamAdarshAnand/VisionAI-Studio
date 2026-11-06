"""Streamlit image processing playground."""
from io import BytesIO
import numpy as np
from PIL import Image, UnidentifiedImageError
import streamlit as st
from vision import MODES, process_image

st.set_page_config(page_title="VisionAI Studio", page_icon="👁️", layout="wide")
st.title("👁️ VisionAI Studio")
st.caption("Explore computer vision interactively with OpenCV. No cloud inference API required.")

with st.sidebar:
    st.header("Processing controls")
    mode = st.selectbox("Operation", MODES, index=3)
    kernel = st.slider("Gaussian kernel", min_value=1, max_value=21, value=5, step=2)
    low = st.slider("Canny low threshold", 0, 255, 60)
    high = st.slider("Canny high threshold", low, 255, max(low, 160))

file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
if file is None:
    st.info("Upload an image to start exploring image-processing operations.")
    st.stop()

try:
    source = Image.open(file).convert("RGB")
    if source.width * source.height > 20_000_000:
        st.error("Image is too large. Use an image under 20 megapixels.")
        st.stop()
    rgb = np.asarray(source)
    output = process_image(rgb, mode, blur_kernel=kernel, canny_low=low, canny_high=high)
except (UnidentifiedImageError, OSError, ValueError) as exc:
    st.error(f"Could not process image: {exc}")
    st.stop()

left, right = st.columns(2)
with left:
    st.subheader("Original")
    st.image(source, use_container_width=True)
with right:
    st.subheader(mode)
    st.image(output, use_container_width=True)

buffer = BytesIO()
Image.fromarray(output).save(buffer, format="PNG")
st.download_button("Download processed PNG", buffer.getvalue(), file_name="visionai_output.png", mime="image/png")
st.caption("Optional YOLOv5-style ONNX inference is available as a developer function in vision.py; model weights are not included.")
