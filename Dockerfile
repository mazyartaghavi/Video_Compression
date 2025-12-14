FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3 python3-pip git cmake \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

COPY src/ ./src/
COPY artifacts/ ./artifacts/

ENV PYTHONPATH=/workspace/src
CMD ["python3", "src/experiments/evaluate.py"]


torch>=2.1
torchvision
numpy
scipy
opencv-python
matplotlib
scikit-image
gymnasium
stable-baselines3
qiskit
qiskit-aer
dwave-ocean-sdk
pyyaml
pandas
