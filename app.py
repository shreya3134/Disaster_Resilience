import os
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import onnxruntime as ort

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

CLASS_NAMES = ["background", "flood", "building", "road", "vegetation"]

# Load ONNX model
session = ort.InferenceSession("floodnet_unet.onnx")
input_name = session.get_inputs()[0].name

@app.route('/')
def index():
    return render_template('index.html', input_img=None, result_img=None, heatmap_img=None)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if not file:
        return redirect(url_for('index'))

    filename = file.filename
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(img_path)

    # Preprocess image
    img = Image.open(img_path).resize((256, 256))
    img_np = np.array(img).astype(np.float32) / 255.0
    img_input = np.expand_dims(img_np, axis=0)  # (1, 256, 256, 3)

    # Predict
    output = session.run(None, {input_name: img_input})[0]
    pred_mask = np.argmax(output[0], axis=-1)  # (256, 256)

    # Dummy GT mask (for illustration)
    gt_mask = np.zeros((256, 256), dtype=int)
    gt_mask[80:180, 80:180] = 1

    # Save segmentation result visualization
    fig, axs = plt.subplots(1, 3, figsize=(20, 8))
    axs[0].imshow(img_np)
    axs[0].set_title("Input Image")
    axs[0].axis("off")

    axs[1].imshow(gt_mask, cmap="tab20", vmin=0, vmax=len(CLASS_NAMES) - 1)
    axs[1].set_title("Ground Truth")
    axs[1].axis("off")

    axs[2].imshow(pred_mask, cmap="tab20", vmin=0, vmax=len(CLASS_NAMES) - 1)
    axs[2].set_title("Predicted Segmentation")
    axs[2].axis("off")

    cmap = plt.cm.get_cmap("tab20", len(CLASS_NAMES))
    norm = plt.Normalize(vmin=0, vmax=len(CLASS_NAMES) - 1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axs.ravel().tolist(), fraction=0.025, pad=0.04, ticks=np.arange(len(CLASS_NAMES)), label='Classes')

    result_filename = f'result_{filename}.png'
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    plt.tight_layout()
    plt.savefig(result_path, dpi=150)
    plt.close()

    # Compute Flood Heatmap
    flood_mask = (pred_mask == 1).astype(int)
    block_size = 16
    h, w = flood_mask.shape
    heatmap_data = np.zeros((h // block_size, w // block_size))

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = flood_mask[i:i + block_size, j:j + block_size]
            avg_flood = block.mean()
            heatmap_data[i // block_size, j // block_size] = avg_flood

    df_heatmap = pd.DataFrame(heatmap_data)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_heatmap, cmap="Reds", cbar_kws={'label': 'Flood Density'})
    plt.title("Flood Intensity Heatmap")
    heatmap_filename = f'heatmap_{filename}.png'
    heatmap_path = os.path.join(app.config['RESULT_FOLDER'], heatmap_filename)
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150)
    plt.close()

    return render_template('index.html',
                           input_img=url_for('static', filename=f"uploads/{filename}"),
                           result_img=url_for('static', filename=f"results/{result_filename}"),
                           heatmap_img=url_for('static', filename=f"results/{heatmap_filename}"),
                           message="Flood Segmentation Completed")

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    app.run(debug=True)
