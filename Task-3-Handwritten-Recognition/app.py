"""
Streamlit application — Handwritten Character Recognition.

A modern UI to recognise handwritten digits either by drawing on a canvas or by
uploading / dragging-and-dropping an image. Shows the prediction, its confidence
and the top-5 candidate classes. Run with:

    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.predict import Predictor  # noqa: E402
from src.utils import RecognitionError  # noqa: E402

st.set_page_config(page_title="Handwriting Recognition", page_icon="✍️", layout="wide")

CUSTOM_CSS = """
<style>
    .big-pred {
        font-size: 5rem; font-weight: 800; text-align: center;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .subtitle { color: #9aa0a6; font-size: 0.95rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model…")
def get_predictor() -> Predictor:
    return Predictor()


def render_result(result: dict) -> None:
    st.markdown(f"<div class='big-pred'>{result['label']}</div>", unsafe_allow_html=True)
    st.metric("Confidence", f"{result['confidence']:.1%}")
    st.markdown("**Top-5 predictions**")
    top = pd.DataFrame(result["top_k"]).rename(columns={"class": "Digit", "probability": "Probability"})
    top = top.set_index("Digit")
    st.bar_chart(top)
    st.dataframe(
        top.assign(Probability=lambda d: (d["Probability"] * 100).round(2).astype(str) + " %"),
        use_container_width=True,
    )


def main() -> None:
    st.title("✍️ Handwritten Character Recognition")
    st.markdown(
        "<p class='subtitle'>A convolutional neural network trained on MNIST. "
        "Draw a digit or upload an image to classify it in real time.</p>",
        unsafe_allow_html=True,
    )

    try:
        predictor = get_predictor()
    except RecognitionError:
        st.error("No trained model found. Run `python -m src.train` first.")
        st.stop()

    tab_draw, tab_upload = st.tabs(["🖌️ Draw", "📤 Upload"])

    with tab_draw:
        try:
            from streamlit_drawable_canvas import st_canvas

            col_canvas, col_result = st.columns([1, 1])
            with col_canvas:
                canvas = st_canvas(
                    fill_color="#000000", stroke_width=18, stroke_color="#FFFFFF",
                    background_color="#000000", height=280, width=280,
                    drawing_mode="freedraw", key="canvas",
                )
            with col_result:
                if canvas.image_data is not None and canvas.image_data[..., :3].sum() > 0:
                    img = Image.fromarray(canvas.image_data.astype("uint8")).convert("L")
                    render_result(predictor.predict(img))
                else:
                    st.info("Draw a digit on the canvas to see the prediction.")
        except ImportError:
            st.warning(
                "Drawing canvas requires `streamlit-drawable-canvas`. "
                "Install it with `pip install streamlit-drawable-canvas`, or use the Upload tab."
            )

    with tab_upload:
        col_up, col_result = st.columns([1, 1])
        with col_up:
            uploaded = st.file_uploader(
                "Drag & drop a digit image", type=["png", "jpg", "jpeg", "bmp"]
            )
            if uploaded is not None:
                image = Image.open(uploaded)
                st.image(image, caption="Input image", width=200)
        with col_result:
            if uploaded is not None:
                render_result(predictor.predict(image))
            else:
                st.info("Upload an image of a single handwritten digit (0–9).")

    with st.expander("About this model"):
        st.write(
            "A VGG-style CNN with batch normalization, dropout and in-graph data "
            "augmentation, trained with early stopping, learning-rate scheduling "
            "and model checkpointing. See the project README for full metrics."
        )


if __name__ == "__main__":
    main()
