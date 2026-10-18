"""Streamlit demo: upload a worksite image -> helmet / no_helmet / person detections + safety status.

Run:  streamlit run app/streamlit_app.py
"""
from collections import Counter
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best.pt"
CLASSES = ["helmet", "no_helmet", "person"]

st.set_page_config(page_title="Safety Helmet Detection", page_icon="🦺", layout="wide")


@st.cache_resource
def load_model(path: str):
    return YOLO(path)


st.title("🦺 Safety Helmet Detection")
st.caption("YOLOv8 model trained to detect helmet, no_helmet and person — AIRI Team PITB AI Internship, Task 1")

if not MODEL_PATH.exists():
    st.error(f"Model not found at {MODEL_PATH}. Copy best.pt from the training notebook into models/.")
    st.stop()

model = load_model(str(MODEL_PATH))
with st.sidebar:
    st.header("Settings")
    conf = st.slider("Confidence threshold", 0.10, 0.90, 0.35, 0.05)
    st.markdown("**Classes**\n- 🟩 helmet\n- 🟥 no_helmet\n- 🟦 person")

files = st.file_uploader("Upload one or more images", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True)
if not files:
    st.info("Upload a worksite photo to check helmet compliance.")
    st.stop()

for f in files:
    image = Image.open(f).convert("RGB")
    result = model.predict(np.array(image), conf=conf, verbose=False)[0]
    counts = Counter(result.names[int(c)] for c in result.boxes.cls.tolist())
    annotated = Image.fromarray(result.plot()[:, :, ::-1])

    st.subheader(f.name)
    if counts.get("no_helmet", 0):
        st.error(f"🚨 VIOLATION — {counts['no_helmet']} worker(s) without a helmet")
    elif counts.get("helmet", 0):
        st.success("✅ COMPLIANT — every detected head wears a helmet")
    else:
        st.warning("ℹ️ No heads detected")

    left, right = st.columns([3, 1])
    left.image(annotated, use_container_width=True)
    for c in CLASSES:
        right.metric(c, counts.get(c, 0))
    rows = [{"class": result.names[int(c)], "confidence": round(float(s), 3),
             "box (x1,y1,x2,y2)": [round(v) for v in b]}
            for c, s, b in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist())]
    if rows:
        right.dataframe(rows, use_container_width=True, hide_index=True)
    buf = BytesIO(); annotated.save(buf, format="JPEG", quality=92)
    right.download_button("Download result", buf.getvalue(), file_name=f"detected_{Path(f.name).stem}.jpg", mime="image/jpeg")
