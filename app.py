"""
TrueFrame AI — Flask Application
Fixes:
  - Auto-delete uploaded video + extracted frames after result is rendered
  - Auto-delete PDF after it is downloaded
  - Threaded cleanup so response is never delayed
"""

from flask import Flask, render_template, request, send_file, after_this_request
import os
import shutil
import threading
import time
from werkzeug.utils import secure_filename
from predict_hybrid import predict_video
from generate_pdf import generate_pdf

app = Flask(__name__)

UPLOAD_FOLDER  = "static/uploads"
RESULTS_FOLDER = "static/results"

for d in [UPLOAD_FOLDER, os.path.join(RESULTS_FOLDER, "trueframe"),
          os.path.join(RESULTS_FOLDER, "deepguard")]:
    os.makedirs(d, exist_ok=True)

app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024   # 200 MB


# ── helpers ──────────────────────────────────────────────────────────────────

def _delete_later(*paths, delay=2):
    """Delete files/dirs in a background thread after `delay` seconds."""
    def _run():
        time.sleep(delay)
        for p in paths:
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                elif os.path.isfile(p):
                    os.remove(p)
            except Exception as e:
                print(f"[CLEANUP] {p}: {e}")
    threading.Thread(target=_run, daemon=True).start()


# ── routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/detect")
def detect():
    return render_template("upload.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return render_template("upload.html", error="No file received.")
    file = request.files["file"]
    if not file.filename:
        return render_template("upload.html", error="No file selected.")

    filename   = secure_filename(file.filename)
    video_name = os.path.splitext(filename)[0]
    filepath   = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    file_size  = os.path.getsize(filepath)

    result = predict_video(filepath, video_name)
    if result is None:
        # Clean up the upload if analysis fails
        _delete_later(filepath)
        return render_template("upload.html",
                               error="Could not process the video. "
                                     "Please check the file and try again.")

    pdf_path = generate_pdf(
        filename  = filename,
        file_size = file_size,
        result    = result,
    )
    result["pdf_filename"] = os.path.basename(pdf_path)

    # Schedule deletion of:
    #   1. The uploaded video
    #   2. The extracted frames folder
    #   3. The PDF  (give extra time so user can still download it)
    frames_dir = os.path.join(RESULTS_FOLDER, "deepguard", video_name)
    _delete_later(filepath, frames_dir, delay=120)   # 2-min window to view
    _delete_later(pdf_path,             delay=300)   # 5-min window to download

    return render_template("result.html",
                           result=result,
                           video=filename)


@app.route("/cleanup/<video_name>", methods=["POST"])
def cleanup(video_name):
    """
    Called by the result page via sendBeacon when the user navigates away.
    Immediately deletes the uploaded video, extracted frames, and PDF for
    that session — no waiting for the 2-min background timer.
    """
    # Sanitise: only allow safe characters (alphanumeric, dash, underscore, dot)
    import re
    if not re.match(r'^[\w.\-]+$', video_name):
        return "", 400

    video_file  = os.path.join(UPLOAD_FOLDER, video_name)
    frames_dir  = os.path.join(RESULTS_FOLDER, "deepguard",
                               os.path.splitext(video_name)[0])
    pdf_file    = os.path.join(RESULTS_FOLDER, "trueframe",
                               os.path.splitext(video_name)[0] + "_report.pdf")

    _delete_later(video_file, frames_dir, pdf_file, delay=0)
    return "", 204


@app.route("/download_pdf/<filename>")
def download_pdf(filename):
    path = os.path.join(RESULTS_FOLDER, "trueframe", filename)
    if not os.path.exists(path):
        return "PDF not found or already deleted.", 404

    @after_this_request
    def _cleanup(response):
        # Delete the PDF immediately after it starts streaming to the client
        _delete_later(path, delay=3)
        return response

    return send_file(path, as_attachment=True,
                     download_name=filename,
                     mimetype="application/pdf")


# if __name__ == "__main__":
#     app.run(debug=False, threaded=True)
#     # app.run(host="0.0.0.0", port=7860)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False, threaded=True)