# MSE Maxwell cavity-mode solver.
#
# Runs as a plain batch/v1 Job (no Spark, no Java). The entrypoint is
# mse_maxwell_solver.py, invoked as:
#
#   python /opt/app/mse_maxwell_solver.py \
#       --job-id <uuid> --units mm --n-index 1.0 --n-modes 20
#
# HTTP(S)_PROXY-style env is not needed; MinIO is reached over the cluster
# network. Credentials arrive as env vars from the mse-minio/mse-db Secrets and
# are never baked into the image.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_RETRIES=20 \
    PIP_NO_CACHE_DIR=1 \
    # The container is allotted 4 CPUs. Cap BLAS/OpenMP so a solve on a bigger
    # node does not oversubscribe it or inflate peak RSS.
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4

# VTK (via pyvista) needs a few shared libraries that python:slim omits.
# python:3.12-slim is Debian, so apt is available at build time; the solver
# itself needs no system packages beyond these.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglx-mesa0 \
        libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app

COPY dockerfiles/maxwell_solver/requirements.txt /opt/app/requirements.txt
RUN pip install --no-cache-dir -r /opt/app/requirements.txt

COPY dockerfiles/maxwell_solver/ /opt/app/
COPY db/ /opt/app/db/

# Run unprivileged: the solver only reads storage and writes results.
RUN useradd --create-home --uid 10001 mse
USER 10001

ENTRYPOINT ["python", "/opt/app/mse_maxwell_solver.py"]
