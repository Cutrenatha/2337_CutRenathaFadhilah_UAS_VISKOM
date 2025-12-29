import gradio as gr
from ultralytics import YOLO
import os
import sys
from pathlib import Path

custom_css = """
body {
    background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
}

.gradio-container {
    font-family: 'Segoe UI', sans-serif;
    max-width: 900px;
    margin: auto;
}

.upload-box {
    background: white;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
}

button {
    border-radius: 12px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}

footer {
    visibility: hidden;
}
"""
# Main App Class
class FruitFreshnessApp:
    def __init__(self, model_path=None):

        if model_path is None:
            model_path = self.find_latest_model()

        if not os.path.exists(model_path):
            print(f"[ERROR] Model tidak ditemukan: {model_path}")
            sys.exit(1)

        print(f"Loading model: {model_path}")
        self.model = YOLO(model_path)

        self.class_info = {
            'freshapples': {'emoji': '🍎', 'name': 'Fresh Apples', 'status': 'FRESH'},
            'freshbanana': {'emoji': '🍌', 'name': 'Fresh Banana', 'status': 'FRESH'},
            'freshoranges': {'emoji': '🍊', 'name': 'Fresh Oranges', 'status': 'FRESH'},
            'rottenapples': {'emoji': '🍎', 'name': 'Rotten Apples', 'status': 'ROTTEN'},
            'rottenbanana': {'emoji': '🍌', 'name': 'Rotten Banana', 'status': 'ROTTEN'},
            'rottenoranges': {'emoji': '🍊', 'name': 'Rotten Oranges', 'status': 'ROTTEN'}
        }

    def find_latest_model(self):
        runs = Path("runs/classify")
        folders = sorted([f for f in runs.iterdir() if f.is_dir() and f.name.startswith("fruit_freshness")])
        if not folders:
            return "runs/classify/fruit_freshness/weights/best.pt"
        return str(folders[-1] / "weights" / "best.pt")

    def predict(self, image):
        if image is None:
            return "❌ Upload image dulu", "", ""

        results = self.model(image, verbose=False)
        probs = results[0].probs

        idx = probs.top1
        conf = probs.top1conf.item()
        label = results[0].names[idx]

        info = self.class_info.get(label, {'emoji': '🍎', 'name': label, 'status': 'UNKNOWN'})
        color = "#4CAF50" if info['status'] == "FRESH" else "#F44336"

        html = f"""
        <div style="text-align:center; padding:25px; background:white;
                    border-radius:18px; box-shadow:0 8px 20px rgba(0,0,0,0.1)">
            <h1 style="font-size:60px">{info['emoji']}</h1>
            <h2>{info['name']}</h2>
            <h3 style="color:{color}">{info['status']}</h3>
            <p><b>Confidence:</b> {conf*100:.2f}%</p>
        </div>
        """

        details = "### 📊 Top Predictions\n\n"
        for i, (iidx, iconf) in enumerate(zip(probs.top5[:3], probs.top5conf[:3]), 1):
            lname = results[0].names[int(iidx)]
            details += f"{i}. **{lname}** → {iconf*100:.2f}%\n"

        tips = (
            "✅ **Aman dikonsumsi**" if info['status'] == "FRESH"
            else "⚠️ **Tidak layak konsumsi**"
        )

        return html, details, tips

    def create_interface(self):
        with gr.Blocks(css=custom_css) as demo:
            gr.Markdown("""
            # 🍎 Fruit Freshness Detection  
            **AI untuk mendeteksi kesegaran buah**
            """)

            with gr.Row():
                with gr.Column(elem_classes="upload-box"):
                    image = gr.Image(type="pil", label="Upload Gambar")
                    btn = gr.Button("🔍 Analyze", variant="primary")

                with gr.Column():
                    out1 = gr.HTML()
                    out2 = gr.Markdown()
                    out3 = gr.Markdown()

            btn.click(self.predict, image, [out1, out2, out3])
        return demo

# Run App
if __name__ == "__main__":
    app = FruitFreshnessApp()
    demo = app.create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
