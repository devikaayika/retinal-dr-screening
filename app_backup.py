
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_image():
    if "retinal_image" not in request.files:
        return "No image was received. Please select an image.", 400

    file = request.files["retinal_image"]

    if file.filename == "":
        return "Please select an image.", 400

    if not allowed_file(file.filename):
        return "Please select a PNG, JPG, or JPEG image.", 400

    filename = secure_filename(file.filename)

    if not filename:
        return "Invalid filename.", 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    return (
        f"Image uploaded successfully: {filename}. "
        "AI prediction will be added in a later step."
    )


if __name__ == "__main__":
    app.run(debug=True)
