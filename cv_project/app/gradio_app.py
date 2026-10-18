"""Gradio demo — runs locally (python app/gradio_app.py) and as a Hugging Face Space.

Hugging Face Spaces: create a Gradio Space, upload this file renamed to app.py,
best.pt, and app/requirements_space.txt renamed to requirements.txt.
"""
from collections import Counter
from pathlib import Path

import gradio as gr
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
CANDIDATES = [HERE / "best.pt", HERE.parent / "models" / "best.pt"]
MODEL_PATH = next((p for p in CANDIDATES if p.exists()), CANDIDATES[-1])
CLASSES = ["helmet", "no_helmet", "person"]
model = YOLO(str(MODEL_PATH))


def detect(image, conf):
    if image is None:
        return None, "Upload an image first", {}
    r = model.predict(image, conf=conf, verbose=False)[0]
    cnt = Counter(r.names[int(c)] for c in r.boxes.cls.tolist())
    if cnt.get("no_helmet", 0):
        status = f"🚨 VIOLATION — {cnt['no_helmet']} worker(s) without a helmet"
    elif cnt.get("helmet", 0):
        status = "✅ COMPLIANT — every detected head wears a helmet"
    else:
        status = "ℹ️ No heads detected"
    return r.plot()[:, :, ::-1], status, {c: cnt.get(c, 0) for c in CLASSES}


demo = gr.Interface(
    fn=detect,
    inputs=[gr.Image(type="numpy", label="Worksite image"),
            gr.Slider(0.1, 0.9, value=0.35, step=0.05, label="Confidence threshold")],
    outputs=[gr.Image(label="Detections"), gr.Textbox(label="Safety status"), gr.JSON(label="Counts")],
    title="🦺 Safety Helmet Detection (YOLOv8)",
    description="Detects helmet, no_helmet and person and flags safety violations. "
                "AIRI Team PITB AI Internship — Task 1, by Hamza Nadeem.",
)

if __name__ == "__main__":
    demo.launch()
