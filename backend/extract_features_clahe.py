import os
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

from main import VPRModel

# -------- CONFIG --------
IMAGE_DIR = "my_dataset/database_images"
CHECKPOINT = "checkpoints/mixvpr.ckpt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- CLAHE FUNCTION --------
def apply_clahe(img):
    img = np.array(img)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    final = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    return Image.fromarray(final)

# -------- MODEL --------

agg_config = {
    "in_channels": 1024,
    "in_h": 20,
    "in_w": 20,
    "out_channels": 1024,
    "mix_depth": 4,
    "mlp_ratio": 1,
    "out_rows": 4
}

model = VPRModel(
    backbone_arch="resnet50",
    layers_to_crop=[4],
    agg_arch="MixVPR",
    agg_config=agg_config
)

state_dict = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

# -------- IMAGE TRANSFORM --------
transform = transforms.Compose([
    transforms.Resize((320,320)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

# -------- FEATURE EXTRACTION --------
descriptors = []
image_names = []

for img_name in sorted(os.listdir(IMAGE_DIR)):

    if not img_name.lower().endswith((".jpg",".png",".jpeg")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)

    img = Image.open(img_path).convert("RGB")

    # APPLY CLAHE
    img = apply_clahe(img)

    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        desc = model(img)

    descriptors.append(desc.cpu().numpy())
    image_names.append(img_name)

descriptors = np.vstack(descriptors)

# -------- SAVE CLAHE DESCRIPTORS --------
np.save("database_descriptors_clahe.npy", descriptors)
np.save("database_image_names_clahe.npy", np.array(image_names))

print("CLAHE FEATURE EXTRACTION DONE")
print("Descriptor shape:", descriptors.shape)