from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pico-algae project CLI wrapper.")
    parser.add_argument("command", nargs="?", choices=["infer", "predict", "train"])
    args, _ = parser.parse_known_args(sys.argv[1:2])
    if args.command is None:
        parser.print_help()
        raise SystemExit(2)

    passthrough = sys.argv[2:]
    root = Path(__file__).resolve().parent

    if args.command == "infer":
        cmd = [sys.executable, "-m", "pico_algae.inference", *passthrough]
        env = dict(**os.environ)
        env["PYTHONPATH"] = str(root / "src") + (f";{env['PYTHONPATH']}" if "PYTHONPATH" in env else "")
        raise SystemExit(subprocess.call(cmd, cwd=root, env=env))

    script_map = {
        "predict": root / "scripts" / "predict_frcnn.py",
        "train": root / "scripts" / "train_frcnn.py",
    }
    script = script_map[args.command]
    cmd = [sys.executable, str(script), *passthrough]
    raise SystemExit(subprocess.call(cmd, cwd=root))


if __name__ == "__main__":
    main()
