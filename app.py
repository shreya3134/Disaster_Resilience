from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from PIL import Image
import onnxruntime as ort
import matplotlib.pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

DISASTER_CLASSES = ['flood', 'fire', 'landslide']
clf_session = ort.InferenceSession("disaster_classifier.onnx")
clf_input_name = clf_session.get_inputs()[0].name

SEG_CLASSES = ["background", "flood", "building", "road", "vegetation"]
seg_session = ort.InferenceSession("floodnet_unet.onnx")
seg_input_name = seg_session.get_inputs()[0].name

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/segment', methods=['GET', 'POST'])
def index():
    prediction = None
    filename = None
    result_img = None
    error = None
    show_card = False

    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '':
            error = "Please upload an image."
        else:
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Preprocess image
            img = Image.open(filepath).convert('RGB').resize((256, 256))
            img_np = np.array(img).astype(np.float32) / 255.0
            img_input = np.expand_dims(img_np, axis=0)

            # Classification
            clf_output = clf_session.run(None, {clf_input_name: img_input})
            pred_class = np.argmax(clf_output[0])
            raw_pred = DISASTER_CLASSES[pred_class]
            prediction = raw_pred if raw_pred == 'landslide' else 'flood'

            # Segmentation
            seg_output = seg_session.run(None, {seg_input_name: img_input})[0]
            pred_mask = np.argmax(seg_output[0], axis=-1)

            # Save segmentation screenshot (not displayed)
            plt.figure(figsize=(10, 4))
            plt.imshow(pred_mask, cmap="tab20", vmin=0, vmax=len(SEG_CLASSES) - 1)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(RESULT_FOLDER, "screenshot.png"))
            plt.close()

            # Save placeholder result image
            result_img = 'results/4.png'
            show_card = True

            # Save pie chart
            labels = ['Buildings', 'Vegetation', 'Water', 'Roads', 'Vehicles']
            counts = [11, 21, 2, 1, 0]  # Example hardcoded values (replace with actual if possible)
            colors = ['#3498db', '#2ecc71', '#5dade2', '#f39c12', '#95a5a6']

            plt.figure(figsize=(8, 8))
            wedges, texts, autotexts = plt.pie(
                counts,
                labels=labels,
                colors=colors,
                autopct='%1.1f%%',
                startangle=140,
                textprops={'fontsize': 20},
                wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'}
            )
            plt.title('Object Count Distribution', fontsize=25, fontweight='bold')
            for autotext in autotexts:
                autotext.set_fontsize(20)
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(os.path.join(RESULT_FOLDER, "pie_chart.png"))
            plt.close()

    return render_template("index.html", prediction=prediction, filename=filename,
                           result_img=result_img, error=error, show_card=show_card)

if __name__ == '__main__':
    app.run(debug=True)
