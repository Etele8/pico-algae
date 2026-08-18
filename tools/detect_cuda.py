"""Pick the right PyTorch wheel channel for this PC, automatically.

Prints exactly one token to stdout:
    cpu | cu126 | cu128 | cu130

It looks at the NVIDIA driver (via `nvidia-smi`, installed with every NVIDIA
driver). `nvidia-smi` reports the *maximum* CUDA version the installed driver
supports; a PyTorch wheel bundles its own CUDA runtime, so it works as long as
the driver supports that CUDA version or newer. We therefore pick the highest
available wheel channel that is <= the driver's supported CUDA version, and fall
back to CPU when there is no NVIDIA GPU (or the driver is too old).

Only the standard library is used, so this runs on any Python before the venv or
PyTorch exist.
"""
import re
import shutil
import subprocess
import sys

# Available PyTorch CUDA wheel channels, highest first, as (value, tag) where
# value = major*100 + minor of the CUDA runtime the wheel needs.
CHANNELS = [
    (1300, "cu130"),
    (1208, "cu128"),
    (1206, "cu126"),
]


def driver_cuda():
    """Return (major, minor) CUDA version the NVIDIA driver supports, or None."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe], capture_output=True, text=True, timeout=20
        ).stdout
    except Exception:
        return None
    m = re.search(r"CUDA Version:\s*([0-9]+)\.([0-9]+)", out)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def pick():
    cu = driver_cuda()
    if not cu:
        return "cpu"
    value = cu[0] * 100 + cu[1]
    for need, tag in CHANNELS:
        if value >= need:
            return tag
    # NVIDIA GPU present but the driver is older than our oldest wheel: use CPU
    # (safe) rather than risk an install that won't run.
    return "cpu"


if __name__ == "__main__":
    sys.stdout.write(pick() + "\n")
