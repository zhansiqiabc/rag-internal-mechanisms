"""Save a reproducible, non-mutating CUDA environment check."""

import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import torch


def version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


record = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "python_executable": sys.executable,
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda_runtime": torch.version.cuda,
    "transformers": version("transformers"),
    "accelerate": version("accelerate"),
    "scikit_learn": version("scikit-learn"),
    "cuda_available": torch.cuda.is_available(),
}
if torch.cuda.is_available():
    record["gpu"] = torch.cuda.get_device_name(0)
    record["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    a = torch.randn(256, 256, device="cuda")
    result = a @ a
    torch.cuda.synchronize()
    record["cuda_matmul_ok"] = bool(torch.isfinite(result).all().item())

path = Path("results/environment.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(record, ensure_ascii=False, indent=2))
