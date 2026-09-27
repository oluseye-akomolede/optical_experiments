# MSE web app: upload an .obj, watch the solver, view the modes.
#
# The app never talks to the solver directly: it writes an upload row and a Job,
# and proxies the result files back out of MinIO. It runs as uid 10001 and needs
# no filesystem beyond its own code.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_RETRIES=20 \
    PIP_NO_CACHE_DIR=1

# No apt packages: the app is pure Python, which keeps the image small and the
# aarch64 build free of a native toolchain.

WORKDIR /opt/app

COPY dockerfiles/web/requirements.txt /opt/app/requirements.txt
RUN pip install --no-cache-dir -r /opt/app/requirements.txt

COPY web/ /opt/app/
COPY db/ /opt/mse/
# The ONE module both images share: the element caps, the unit table and the
# sizing functions. The app needs them (it sizes the upload and routes the job)
# and must NOT import the solver package to get them -- that package is not in
# this image, and importing it here is exactly what crashlooped the dispatch
# deploy: `/opt/app/config.py:112 ModuleNotFoundError: No module named
# 'mse_maxwell_solver'`. mse_common imports only numpy, so the web image gains
# nothing but a 20 KB file, and there is still exactly one definition of each
# constant for both images to read.
COPY dockerfiles/maxwell_solver/mse_common.py /opt/app/
# The app adds this directory to sys.path for a checkout; in the image the
# module is already here, so name it explicitly rather than relying on the
# interpreter's default path.
ENV PYTHONPATH=/opt/app

# The DDL lives at a fixed path in the image rather than being found by a
# relative climb from the module, which is what made the first deploy fragile.
# DB code reads this from the environment (see web/db.py: MSE_SCHEMA_PATH is the
# first candidate), so the layout is declared here, once, and the resolver's
# fallbacks still cover a bare checkout.
ENV MSE_SCHEMA_PATH=/opt/mse/001_schema.sql

RUN useradd --create-home --uid 10001 mse
USER 10001

EXPOSE 5000

# gunicorn, not the Flask dev server. EXACTLY ONE worker: the background ticker
# that launches solver Jobs runs in-process, and two workers would race for the
# single namespace Job slot. Threads give the concurrency instead, which is safe
# because the ticker is guarded and every other request is I/O bound.
CMD ["gunicorn", "-w", "1", "--threads", "8", "-b", "0.0.0.0:5000", "app:build_app()"]
