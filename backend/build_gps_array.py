import numpy as np
import pandas as pd

# Load image names (same order as descriptors)
image_names = np.load("database_image_names.npy")

# Load CSV
df = pd.read_csv("image_gps_data.csv")

# Build lookup table
gps_dict = {}
for _, row in df.iterrows():
    gps_dict[row["filename"]] = [row["latitude"], row["longitude"]]

# Build GPS array in descriptor order
gps_coords = []
missing = []

for name in image_names:
    if name in gps_dict:
        gps_coords.append(gps_dict[name])
    else:
        missing.append(name)

gps_coords = np.array(gps_coords, dtype=np.float32)

# Save
np.save("gps_coords.npy", gps_coords)

print("GPS shape:", gps_coords.shape)

if missing:
    print("Missing GPS for:", missing)
else:
    print("All images matched successfully")
