FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace/pico-algae

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements-dev.txt ./requirements-dev.txt
COPY pyproject.toml ./pyproject.toml
COPY setup.cfg* setup.py* README* MANIFEST.in* ./
COPY src ./src

RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements-dev.txt

COPY . .

ENV PYTHONPATH=/workspace/pico-algae
CMD ["bash"]