from flask import Flask, request, render_template, redirect, url_for, flash
import boto3
import trimesh
import tempfile
import os
from werkzeug.utils import secure_filename

# --- Configuration ---
BUCKET_NAME = "mse-meshes"
MINIO_ENDPOINT = "https://minio.taildfdd3b.ts.net"
ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")

# --- Boto3 client ---
s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
)

# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "stlfile" not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files["stlfile"]
        filename = secure_filename(file.filename)

        if not filename.lower().endswith(".stl"):
            flash("Only .stl files are allowed")
            return redirect(request.url)

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.stl') as tmp:
                file.save(tmp.name)
                mesh = trimesh.load(tmp.name)

                if not isinstance(mesh, trimesh.Trimesh) or mesh.faces.shape[1] != 3:
                    flash("STL file is not a valid triangulated mesh")
                    os.unlink(tmp.name)
                    return redirect(request.url)

                s3.upload_file(tmp.name, BUCKET_NAME, filename)
                flash(f"Successfully uploaded {filename}")
                os.unlink(tmp.name)

        except Exception as e:
            flash(f"Failed to upload file: {e}")
            return redirect(request.url)

    return render_template("upload.html")