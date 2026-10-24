
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Use the same dataset location as check_dataset.py
DATASET_DIR = Path(
    r"C:\Users\hp\Downloads\aptos2019-blindness-detection"
)

CSV_PATH = DATASET_DIR / "train.csv"
IMAGE_DIR = DATASET_DIR / "train_images"

# Load image IDs and severity labels
df = pd.read_csv(CSV_PATH)

# Check the expected CSV columns
required_columns = {"id_code", "diagnosis"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        "train.csv must contain id_code and diagnosis columns."
    )

# Build image paths and check that every image exists
df["image_path"] = df["id_code"].apply(
    lambda image_id: str(IMAGE_DIR / f"{image_id}.png")
)

missing_images = [
    path for path in df["image_path"]
    if not Path(path).is_file()
]

if missing_images:
    raise FileNotFoundError(
        f"{len(missing_images)} images were not found. "
        "Check the image folder and file extensions."
    )

# Make sure labels are in the expected range
if not df["diagnosis"].isin([0, 1, 2, 3, 4]).all():
    raise ValueError("Unexpected severity labels found.")

# Split while preserving class proportions
train_df, val_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["diagnosis"]
)

# Save the lists inside the project
output_dir = Path("dataset")
output_dir.mkdir(exist_ok=True)

train_df.to_csv(output_dir / "train_split.csv", index=False)
val_df.to_csv(output_dir / "val_split.csv", index=False)

print("Dataset preparation check completed!")
print("Total images:", len(df))
print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Missing images:", len(missing_images))
print("\nTraining class counts:")
print(train_df["diagnosis"].value_counts().sort_index())
print("\nValidation class counts:")
print(val_df["diagnosis"].value_counts().sort_index())
