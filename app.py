from ultralytics import YOLO
import gradio as gr
from PIL import Image

# Load YOLOv8 model
model = YOLO("best.pt")

def detect(image, conf):
    results = model(image, conf=conf)
    annotated = results[0].plot()
    return Image.fromarray(annotated)

css = """
body {
    background-color: #0f172a;
}
h1, p {
    color: #e5e7eb;
    text-align: center;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("<h1>🧠 Tumor Detection</h1>")
    gr.Markdown("<p>Upload an image and detect objects</p>")

    with gr.Row():
        img = gr.Image(type="pil", label="Upload Image")
        out = gr.Image(label="Detection Result")

    conf = gr.Slider(0.1, 1.0, value=0.5, label="Confidence Threshold")

    btn = gr.Button("Run Detection")
    btn.click(fn=detect, inputs=[img, conf], outputs=out)

demo.launch()

##### url- http://127.0.0.1:7861/
