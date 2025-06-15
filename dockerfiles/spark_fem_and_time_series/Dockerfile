FROM python:3.12-slim

# Speed up pip
ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_RETRIES=20

    

# Install required Python libraries (combine into one layer)
RUN pip install --no-cache-dir \
    pyspark==3.5.1 \
    pyvista \
    tetgen \
    scipy \
    numpy \
    numba \
    scikit-image \
    torch \
    torchvision \
    torchaudio \
    mlflow \
    xgboost \
    scikit-learn \
    seaborn \
    matplotlib

# Install Java for Spark (OpenJDK 11)
USER root
# Install Java 17 and dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set Java path
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"
RUN java -version

# Copy the Spark job into the image
COPY spark-fem-solver.py /opt/app/spark-fem-solver.py

# Make sure it’s executable (optional if using ENTRYPOINT)
RUN chmod +x /opt/app/spark-fem-solver.py

# Revert to notebook user for better security
USER $NB_UID

# Set default entrypoint to run the spark job with --stl argument
ENTRYPOINT ["spark-submit", "/opt/app/spark-fem-solver.py"]
