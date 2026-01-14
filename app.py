from ultralytics import YOLO
import gradio as gr
from PIL import Image

# Load YOLOv8 model
model = YOLO("best.pt")

def detect(image, conf):
    if image is None:
        return None
    results = model(image, conf=conf)
    annotated = results[0].plot()
    return Image.fromarray(annotated)

# ✅ Minimal, safe CSS (Gradio-friendly)
css = """
body {
    background:
        linear-gradient(
            rgba(255, 255, 255, 0.88),
            rgba(255, 255, 255, 0.88)
        ),
        url("mri_background.png")
        center / cover no-repeat fixed;
}

.gradio-container {
    max-width: 1000px;
    margin: auto;
    background: white;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.12);
}

h1 {
    text-align: center;
    color: #111;
}

.subtitle {
    text-align: center;
    color: #555;
    margin-bottom: 20px;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# 🧠 Brain Tumor Detection")
    gr.Markdown(
        "<div class='subtitle'>"
        "Upload a brain MRI image to detect tumor regions using an AI model."
        "</div>"
    )

    with gr.Row():
        img = gr.Image(type="pil", label="Upload MRI Image")
        out = gr.Image(label="Detection Result")

    conf = gr.Slider(
        minimum=0.1,
        maximum=1.0,
        value=0.5,
        label="Confidence Threshold"
    )

    gr.Button("Run Detection").click(
        fn=detect,
        inputs=[img, conf],
        outputs=out
    )

demo.launch()
