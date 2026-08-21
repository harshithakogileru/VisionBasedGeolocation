import os
import numpy as np
import torch
import faiss
import cv2
from PIL import Image
from torchvision import transforms
from main import VPRModel

# ================= CONFIG =================

DB_DESC_FILE = "database_descriptors_clahe.npy"
DB_NAMES_FILE = "database_image_names_clahe.npy"
QUERY_DIR = "query_images"
CHECKPOINT = "checkpoints/mixvpr.ckpt"

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


# ================= LOAD DATABASE =================

db_desc = np.load(DB_DESC_FILE).astype("float32")

db_names = np.load(DB_NAMES_FILE)

faiss.normalize_L2(db_desc)

index = faiss.IndexFlatIP(db_desc.shape[1])

index.add(db_desc)


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
    transforms.Resize((320,320)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


# ================= METRICS =================

recall1 = 0
recall2 = 0
recall3 = 0

total_queries = 0


# ================= LOOP OVER QUERY IMAGES =================

for q_name in sorted(os.listdir(QUERY_DIR)):

    if not q_name.lower().endswith((".jpg",".png",".jpeg")):
        continue

    q_path = os.path.join(QUERY_DIR, q_name)

    img = Image.open(q_path).convert("RGB")

    # apply CLAHE
    img = apply_clahe(img)

    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        q_desc = model(img).cpu().numpy().astype("float32")

    faiss.normalize_L2(q_desc)

    similarities, indices = index.search(q_desc, 3)

    retrieved = db_names[indices[0]]

    query_loc = q_name.split("_")[0]

    r1 = retrieved[0].split("_")[0]
    r2 = retrieved[1].split("_")[0]
    r3 = retrieved[2].split("_")[0]

    print("\nQuery:", q_name)
    print("Top 3:", retrieved)

    if query_loc == r1:
        recall1 += 1
        recall2 += 1
        recall3 += 1

    elif query_loc == r2:
        recall2 += 1
        recall3 += 1

    elif query_loc == r3:
        recall3 += 1

    total_queries += 1


# ================= FINAL RESULTS =================

print("\n===== FINAL RESULTS =====")

print("Total Queries:", total_queries)

print(f"Recall@1: {(recall1 / total_queries) * 100:.2f}")
print(f"Recall@2: {(recall2 / total_queries) * 100:.2f}")
print(f"Recall@3: {(recall3 / total_queries) * 100:.2f}")
