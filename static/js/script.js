
document.addEventListener("DOMContentLoaded", function () {
    const imageInput = document.getElementById("retinal-image");
    const imagePreview = document.getElementById("image-preview");
    const imageInfo = document.getElementById("image-info");
    const clearButton = document.getElementById("clear-button");
    const uploadForm = document.getElementById("upload-form");
    const uploadButton = document.getElementById("upload-button");

    // Set the prediction score bar.
    // This works with data-score="{{ prediction.confidence }}"
    // in index.html.
    const scoreFill = document.querySelector(".score-fill");

    if (scoreFill) {
        const score = Number(scoreFill.dataset.score);

        if (Number.isFinite(score)) {
            const safeScore = Math.max(0, Math.min(100, score));
            scoreFill.style.width = safeScore + "%";
        }
    }

    // Stop safely if the upload form elements are missing.
    if (
        !imageInput ||
        !imagePreview ||
        !imageInfo ||
        !clearButton ||
        !uploadForm ||
        !uploadButton
    ) {
        return;
    }

    const MAX_FILE_SIZE = 10 * 1024 * 1024;
    const ALLOWED_TYPES = ["image/png", "image/jpeg"];

    let previewUrl = null;

    // Clear the image preview and release its temporary URL.
    function clearPreview() {
        imagePreview.hidden = true;
        imagePreview.removeAttribute("src");

        if (previewUrl) {
            URL.revokeObjectURL(previewUrl);
            previewUrl = null;
        }
    }

    // Restore the upload button.
    function resetSubmitButton() {
        uploadButton.disabled = false;
        uploadButton.textContent = "Upload and Predict";
    }

    // Preview and validate the selected image.
    imageInput.addEventListener("change", function () {
        const file = imageInput.files[0];

        clearPreview();
        resetSubmitButton();

        if (!file) {
            imageInfo.textContent = "No image selected yet.";
            return;
        }

        if (!ALLOWED_TYPES.includes(file.type)) {
            imageInfo.textContent =
                "Please select a PNG, JPG, or JPEG image.";
            imageInput.value = "";
            return;
        }

        if (file.size === 0) {
            imageInfo.textContent =
                "The selected file is empty.";
            imageInput.value = "";
            return;
        }

        if (file.size > MAX_FILE_SIZE) {
            imageInfo.textContent =
                "Image exceeds the 10 MB size limit.";
            imageInput.value = "";
            return;
        }

        previewUrl = URL.createObjectURL(file);
        imagePreview.src = previewUrl;
        imagePreview.hidden = false;

        const sizeInKB = (file.size / 1024).toFixed(2);

        imageInfo.textContent =
            `Selected: ${file.name} | Size: ${sizeInKB} KB`;
    });

    // Clear the selected image.
    clearButton.addEventListener("click", function () {
        imageInput.value = "";
        clearPreview();
        resetSubmitButton();

        imageInfo.textContent = "No image selected yet.";
    });

    // Validate before submitting the image to Flask.
    uploadForm.addEventListener("submit", function (event) {
        const file = imageInput.files[0];

        if (!file) {
            event.preventDefault();
            imageInfo.textContent =
                "Please select an image first.";
            return;
        }

        if (
            !ALLOWED_TYPES.includes(file.type) ||
            file.size === 0 ||
            file.size > MAX_FILE_SIZE
        ) {
            event.preventDefault();
            imageInfo.textContent =
                "Select a valid PNG, JPG, or JPEG image under 10 MB.";
            return;
        }

        uploadButton.disabled = true;
        uploadButton.textContent = "Analyzing Image...";

        imageInfo.textContent =
            "Image submitted. Please wait for the prediction...";
    });

    // Scroll to the results after Flask returns a prediction.
    const results = document.getElementById("prediction-results");

    if (results) {
        results.setAttribute("tabindex", "-1");

        window.setTimeout(function () {
            results.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });
        }, 150);
    }
});
