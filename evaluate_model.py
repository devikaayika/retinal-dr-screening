
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

PROJECT_DIR = Path(__file__).resolve().parent
VAL_CSV = PROJECT_DIR / "dataset" / "val_split.csv"
MODEL_PATH = PROJECT_DIR / "model" / "best_retinal_model.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

# Load validation records and the trained checkpoint
val_df = pd.read_csv(VAL_CSV)
model = tf.keras.models.load_model(MODEL_PATH)

def load_image(image_path, label):
    image_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_png(image_bytes, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    return image, tf.cast(label, tf.int32)

val_ds = tf.data.Dataset.from_tensor_slices(
    (
        val_df["image_path"].astype(str).values,
        val_df["diagnosis"].astype("int32").values
    )
)

val_ds = val_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Predict each validation image
y_true = val_df["diagnosis"].astype(int).to_numpy()
probabilities = model.predict(val_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

print("\nValidation accuracy:",
      round(accuracy_score(y_true, y_pred), 4))

print("Macro F1-score:",
      round(f1_score(y_true, y_pred, average="macro",
                     zero_division=0), 4))

print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1, 2, 3, 4],
    target_names=CLASS_NAMES,
    zero_division=0
))

print("\nConfusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4]))

# Save detailed predictions for later review
results = val_df[["id_code", "diagnosis"]].copy()
results["predicted_class"] = y_pred
results["confidence"] = probabilities.max(axis=1)

report_path = PROJECT_DIR / "reports" / "validation_predictions.csv"
report_path.parent.mkdir(exist_ok=True)
results.to_csv(report_path, index=False)

print("\nSaved validation predictions to:", report_path)
