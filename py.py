import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Define class names (adjust as per your dataset)
CLASS_NAMES = ["background", "flood", "building", "road", "vegetation"]

# Load ONNX model
session = ort.InferenceSession("floodnet_unet.onnx")
input_name = session.get_inputs()[0].name

# Load and preprocess the input image
img = Image.open("6279.jpg").resize((256, 256))
img_np = np.array(img).astype(np.float32) / 255.0
img_input = np.expand_dims(img_np, axis=0)  # (1, 256, 256, 3)

# Run inference
output = session.run(None, {input_name: img_input})[0]
pred_mask = np.argmax(output[0], axis=-1)  # (256, 256)

# Dummy ground truth for demo (optional)
gt_mask = np.zeros((256, 256), dtype=int)
gt_mask[80:180, 80:180] = 1  # Simulated flood region

# Visualization
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(img_np)
axs[0].set_title("Input Image")
axs[0].axis("off")

axs[1].imshow(gt_mask, cmap="tab20", vmin=0, vmax=len(CLASS_NAMES)-1)
axs[1].set_title("Ground Truth (Simulated)")
axs[1].axis("off")

axs[2].imshow(pred_mask, cmap="tab20", vmin=0, vmax=len(CLASS_NAMES)-1)
axs[2].set_title("Prediction")
axs[2].axis("off")

# Colorbar
cmap = plt.cm.get_cmap("tab20", len(CLASS_NAMES))
norm = plt.Normalize(vmin=0, vmax=len(CLASS_NAMES)-1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), fraction=0.02, pad=0.04)
cbar.set_ticks(np.arange(len(CLASS_NAMES)))
cbar.set_ticklabels(CLASS_NAMES)

plt.tight_layout()
plt.show()