from fastapi import FastAPI, UploadFile, File
import numpy as np
import torch
import faiss
from PIL import Image
from torchvision import transforms
from main import VPRModel
import io

# ================= CONFIG =================
DB_DESCRIPTORS = "database_descriptors.npy"
DB_IMAGE_NAMES = "database_image_names.npy"
GPS_FILE = "gps_coords.npy"
CHECKPOINT = "checkpoints/mixvpr.ckpt"

TOP_K = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

# ================= LOAD DATABASE ONCE =================
descriptors = np.load(DB_DESCRIPTORS).astype("float32")
image_names = np.load(DB_IMAGE_NAMES)
gps_coords = np.load(GPS_FILE)

faiss.normalize_L2(descriptors)

index = faiss.IndexFlatIP(descriptors.shape[1])
index.add(descriptors)

# ================= LOAD MODEL ONCE =================
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

# ================= IMAGE TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================= API ENDPOINT =================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        q_desc = model(img).cpu().numpy().astype("float32")

    faiss.normalize_L2(q_desc)

    similarities, indices = index.search(q_desc, TOP_K)

    topk_gps = []
    weights = []

    for idx, sim in zip(indices[0], similarities[0]):
        lat, lon = gps_coords[idx]
        topk_gps.append([lat, lon])
        weights.append(sim)

    topk_gps = np.array(topk_gps)
    weights = np.array(weights)

    weights /= weights.sum()

    estimated_location = (topk_gps * weights[:, None]).sum(axis=0)

    return {
        "latitude": float(estimated_location[0]),
        "longitude": float(estimated_location[1])
    }
