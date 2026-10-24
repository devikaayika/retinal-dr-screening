from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify
)
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

from database import (
    initialize_database,
    save_prediction,
    get_prediction_history
)


# --------------------------------------------------
# Application configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
app.secret_key = "change-this-to-a-random-secret"

MODEL_PATH = BASE_DIR / "model" / "best_retinal_model.keras"
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "No Diabetic Retinopathy",
    "Mild Diabetic Retinopathy",
    "Moderate Diabetic Retinopathy",
    "Severe Diabetic Retinopathy",
    "Proliferative Diabetic Retinopathy"
]

CLASS_SHORT_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]


# --------------------------------------------------
# Initialize database and load trained model
# --------------------------------------------------

initialize_database()

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

model = load_model(str(MODEL_PATH))


# --------------------------------------------------
# Image utilities
# --------------------------------------------------

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def prepare_image(image_path):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE)

        image_array = np.asarray(image, dtype=np.float32)

    # Keep pixel values in the 0–255 range.
    # This must match the preprocessing used during training.
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict_retinal_image(image_path):
    image_array = prepare_image(image_path)

    predictions = np.asarray(
        model.predict(image_array, verbose=0)
    )

    if predictions.ndim != 2 or predictions.shape[0] != 1:
        raise ValueError("Unexpected model output shape.")

    scores = predictions[0]

    if len(scores) != len(CLASS_NAMES):
        raise ValueError(
            "The model output does not contain five classes."
        )

    if not np.all(np.isfinite(scores)):
        raise ValueError("The model returned invalid scores.")

    class_id = int(np.argmax(scores))
    confidence = float(scores[class_id]) * 100

    class_scores = []

    for index, score in enumerate(scores):
        class_scores.append({
            "name": CLASS_NAMES[index],
            "short_name": CLASS_SHORT_NAMES[index],
            "score": float(score) * 100
        })

    return {
        "class_id": class_id,
        "predicted_class": CLASS_NAMES[class_id],
        "short_name": CLASS_SHORT_NAMES[class_id],
        "confidence": confidence,
        "class_scores": class_scores
    }


# --------------------------------------------------
# Dashboard statistics
# --------------------------------------------------

def get_dashboard_stats():
    records = get_prediction_history()

    today = __import__("datetime").datetime.now().date().isoformat()

    distribution = [0] * len(CLASS_NAMES)
    today_count = 0

    for record in records:
        class_id = int(record["class_id"])

        if 0 <= class_id < len(distribution):
            distribution[class_id] += 1

        if str(record["created_at"]).startswith(today):
            today_count += 1

    recent_records = records[:10]

    for record in recent_records:
        record["short_name"] = CLASS_SHORT_NAMES[
            int(record["class_id"])
        ]

    return {
        "total": len(records),
        "today": today_count,
        "distribution": distribution,
        "daily_labels": [
            record["created_at"][:10]
            for record in reversed(records[-7:])
        ],
        "daily_counts": list(range(1, len(records[-7:]) + 1)),
        "recent_records": recent_records
    }


# --------------------------------------------------
# Dashboard
# --------------------------------------------------

@app.route("/")
def index():
    stats = get_dashboard_stats()

    return render_template(
        "index.html",
        stats=stats,
        recent_records=stats["recent_records"],
        result=None
    )


# --------------------------------------------------
# Upload image and predict
# --------------------------------------------------

@app.route("/upload", methods=["POST"])
def upload_image():
    if "retinal_image" not in request.files:
        flash("Please select a retinal image.", "error")
        return redirect(url_for("index"))

    file = request.files["retinal_image"]

    if not file or file.filename == "":
        flash("Please select an image before uploading.", "error")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Upload a PNG, JPG, or JPEG image.", "error")
        return redirect(url_for("index"))

    original_name = secure_filename(file.filename)

    if not original_name:
        flash("Invalid filename.", "error")
        return redirect(url_for("index"))

    # Use a unique filename so uploads do not overwrite each other.
    import uuid

    extension = original_name.rsplit(".", 1)[1].lower()
    saved_name = f"{uuid.uuid4().hex}.{extension}"
    image_path = UPLOAD_FOLDER / saved_name

    try:
        file.save(image_path)

        # Validate that the uploaded file is a real image.
        with Image.open(image_path) as image:
            image.verify()

        prediction = predict_retinal_image(image_path)

        save_prediction(
            image_filename=saved_name,
            predicted_class=prediction["predicted_class"],
            class_id=prediction["class_id"],
            confidence=prediction["confidence"]
        )

        result = {
            **prediction,
            "image_filename": saved_name,
            "image_url": url_for(
                "static",
                filename=f"uploads/{saved_name}"
            )
        }

        stats = get_dashboard_stats()

        return render_template(
            "index.html",
            stats=stats,
            recent_records=stats["recent_records"],
            result=result
        )

    except (UnidentifiedImageError, OSError):
        if image_path.exists():
            image_path.unlink()

        flash("The uploaded file is not a valid image.", "error")
        return redirect(url_for("index"))

    except Exception:
        app.logger.exception("Image prediction failed")

        if image_path.exists():
            image_path.unlink()

        flash(
            "Prediction failed. Check the model and server logs.",
            "error"
        )
        return redirect(url_for("index"))


# --------------------------------------------------
# Prediction history
# --------------------------------------------------

@app.route("/history")
def history():
    records = get_prediction_history()

    for record in records:
        class_id = int(record["class_id"])
        record["short_name"] = (
            CLASS_SHORT_NAMES[class_id]
            if 0 <= class_id < len(CLASS_SHORT_NAMES)
            else "Unknown"
        )

    return render_template(
        "history.html",
        records=records
    )


# --------------------------------------------------
# Dashboard API
# --------------------------------------------------

@app.route("/api/stats")
def api_stats():
    stats = get_dashboard_stats()

    recent_predictions = []

    for record in stats["recent_records"]:
        item = dict(record)
        item["predicted_class"] = item["short_name"]
        item["image_url"] = url_for(
            "static",
            filename=f"uploads/{item['image_filename']}"
        )
        recent_predictions.append(item)

    distribution = stats["distribution"]

    return jsonify({
        "total_predictions": stats["total"],
        "today_predictions": stats["today"],
        "no_dr_count": distribution[0],
        "dr_count": sum(distribution[1:]),
        "latest_confidence": (
            recent_predictions[0]["confidence"]
            if recent_predictions else None
        ),
        "chart_labels": CLASS_SHORT_NAMES,
        "chart_values": distribution,
        "recent_predictions": recent_predictions
    })


# --------------------------------------------------
# Error handling
# --------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):
    flash("Image is too large. Maximum size is 10 MB.", "error")
    return redirect(url_for("index")), 303


if __name__ == "__main__":
    app.run(debug=True)