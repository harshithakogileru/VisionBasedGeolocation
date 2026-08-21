import os
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

# --------- CONFIG ---------
IMAGE_FOLDER = "my_dataset/database_images"   # path to your images folder
OUTPUT_CSV = "image_gps_data.csv"
# --------------------------

def get_exif_data(image):
    exif_data = {}
    info = image._getexif()
    if info is None:
        return exif_data

    for tag, value in info.items():
        decoded = TAGS.get(tag, tag)
        if decoded == "GPSInfo":
            gps_data = {}
            for t in value:
                sub_decoded = GPSTAGS.get(t, t)
                gps_data[sub_decoded] = value[t]
            exif_data["GPSInfo"] = gps_data
        else:
            exif_data[decoded] = value

    return exif_data


def convert_to_degrees(value):
    d, m, s = value
    return float(d) + float(m) / 60 + float(s) / 3600


def get_lat_lon(exif_data):
    if "GPSInfo" not in exif_data:
        return None, None

    gps_info = exif_data["GPSInfo"]

    try:
        lat = convert_to_degrees(gps_info["GPSLatitude"])
        if gps_info["GPSLatitudeRef"] != "N":
            lat = -lat

        lon = convert_to_degrees(gps_info["GPSLongitude"])
        if gps_info["GPSLongitudeRef"] != "E":
            lon = -lon

        return lat, lon
    except:
        return None, None


rows = []

for filename in os.listdir(IMAGE_FOLDER):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(IMAGE_FOLDER, filename)

        try:
            img = Image.open(img_path)
            exif_data = get_exif_data(img)
            lat, lon = get_lat_lon(exif_data)

            if lat is not None and lon is not None:
                rows.append({
                    "filename": filename,
                    "latitude": lat,
                    "longitude": lon
                })
            else:
                print(f"No GPS data: {filename}")

        except Exception as e:
            print(f"Error reading {filename}: {e}")

# Save to CSV
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved GPS data to {OUTPUT_CSV}")
