
import pandas as pd
from pathlib import Path
from PIL import Image

DATASET_DIR = Path(
    r"C:\Users\hp\Downloads\aptos2019-blindness-detection"
)

train_csv = DATASET_DIR / "train.csv"
train_images = DATASET_DIR / "train_images"

df = pd.read_csv(train_csv)

# Select the first image listed in train.csv
image_id = df.iloc[0]["id_code"]
label = df.iloc[0]["diagnosis"]

# APTOS image files commonly use .png extensions
image_path = train_images / f"{image_id}.png"

if not image_path.exists():
    print("Image not found:", image_path)
else:
    with Image.open(image_path) as image:
        print("Image loaded successfully!")
        print("Image ID:", image_id)
        print("Diagnosis label:", label)
        print("Image dimensions:", image.size)
        print("Image format:", image.format)
        image.show()
