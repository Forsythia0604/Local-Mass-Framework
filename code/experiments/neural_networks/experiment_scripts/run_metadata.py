"""Write a JSON record beside newly generated marginal arrays."""

from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys


def write_metadata(output, args, **details):
    path = Path(output)
    if path.suffix != ".npz":
        path = Path(str(path) + ".npz")
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "arguments": vars(args),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in ("numpy", "torch", "torchvision")},
        "npz_file": path.name,
        "npz_sha256": sha256(path.read_bytes()).hexdigest(),
        "details": details,
    }
    path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )
