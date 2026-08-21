import sys
import faiss
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms
from main import VPRModel

# ================= CONFIG =================
DB_DESCRIPTORS = "database_descriptors_clahe.npy"
DB_IMAGE_NAMES = "database_image_names_clahe.npy"
GPS_FILE = "gps_coords.npy"
CHECKPOINT = "checkpoints/mixvpr.ckpt"

TOP_K = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= CLAHE FUNCTION =================
def apply_clahe(img):

    img = np.array(img)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    final = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    return Image.fromarray(final)

# ================= INPUT =================
if len(sys.argv) != 2:
    print("Usage: python localize_query_clahe.py <query_image_path>")
    sys.exit(1)

QUERY_IMAGE = sys.argv[1]

# ================= LOAD DATABASE =================
descriptors = np.load(DB_DESCRIPTORS).astype("float32")
image_names = np.load(DB_IMAGE_NAMES)
gps_coords = np.load(GPS_FILE)

# Normalize database descriptors
faiss.normalize_L2(descriptors)

# ================= FAISS INDEX =================
index = faiss.IndexFlatIP(descriptors.shape[1])
index.add(descriptors)

# ================= MODEL =================
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
        std =[0.229, 0.224, 0.225]
    )
])

# ================= QUERY DESCRIPTOR =================
img = Image.open(QUERY_IMAGE).convert("RGB")

# APPLY CLAHE
img = apply_clahe(img)

img = transform(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    q_desc = model(img).cpu().numpy().astype("float32")

# Normalize query descriptor
faiss.normalize_L2(q_desc)

# ================= FAISS SEARCH =================
similarities, indices = index.search(q_desc, TOP_K)

# ================= OUTPUT =================
print(f"\nTop-{TOP_K} matches for: {QUERY_IMAGE}\n")

topk_gps = []
weights = []

for rank,(idx,sim) in enumerate(zip(indices[0], similarities[0]), start=1):

    name = image_names[idx]
    lat, lon = gps_coords[idx]

    print(
        f"{rank}. index={idx:<3} "
        f"file={name:<22} "
        f"similarity={sim:.4f} "
        f"lat={lat:.7f} "
        f"lon={lon:.7f}"
    )

    topk_gps.append([lat,lon])
    weights.append(sim)

# ================= WEIGHTED GPS =================
topk_gps = np.array(topk_gps)
weights = np.array(weights)

weights /= weights.sum()

estimated_location = (topk_gps * weights[:,None]).sum(axis=0)

print(
    f"\nEstimated location (weighted avg of top-{TOP_K}): "
    f"{estimated_location[0]}, {estimated_location[1]}"
)