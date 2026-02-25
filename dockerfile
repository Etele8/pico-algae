FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace/pico-algae

# Base deps + SSH server
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates libgl1 libglib2.0-0 \
    openssh-server \
 && rm -rf /var/lib/apt/lists/*

# SSHD setup (key-only auth)
RUN mkdir -p /var/run/sshd && \
    sed -i 's/#Port 22/Port 22/' /etc/ssh/sshd_config && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config || true && \
    printf "\nClientAliveInterval 60\nClientAliveCountMax 120\n" >> /etc/ssh/sshd_config

# create /root/.ssh so RunPod key injection has a target
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

COPY requirements-dev.txt ./requirements-dev.txt
COPY pyproject.toml ./pyproject.toml
COPY setup.cfg* setup.py* README* MANIFEST.in* ./
COPY src ./src

RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements-dev.txt

COPY . .

ENV PYTHONPATH=/workspace/pico-algae

# Expose SSH
EXPOSE 22

# Start sshd + keep container alive
CMD ["bash", "-lc", "service ssh start && tail -f /dev/null"]