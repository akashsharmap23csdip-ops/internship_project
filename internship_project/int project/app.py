from __future__ import annotations

import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from analysis_pipeline import run_analysis


ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"
OUTPUTS_DIR = ROOT_DIR / "outputs"
UPLOAD_DIR = ROOT_DIR / "data" / "uploads"
PORT = 8000
ALLOWED_EXTENSIONS = {".csv"}

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="/frontend")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


def _is_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index() -> object:
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/outputs/<path:filename>")
def outputs(filename: str) -> object:
    return send_from_directory(OUTPUTS_DIR, filename)


@app.route("/api/analyze", methods=["POST"])
def analyze() -> object:
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    upload = request.files["file"]
    if not upload or not upload.filename:
        return jsonify({"error": "No file selected."}), 400

    if not _is_allowed(upload.filename):
        return jsonify({"error": "Only CSV files are supported."}), 400

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = secure_filename(upload.filename)
    saved_path = UPLOAD_DIR / f"{int(time.time())}_{safe_name}"
    upload.save(saved_path)

    try:
        result = run_analysis(saved_path, OUTPUTS_DIR)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        print(f"Analysis failed: {exc}")
        return jsonify({"error": "Analysis failed on the server."}), 500

    result["plots_version"] = int(time.time())
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)