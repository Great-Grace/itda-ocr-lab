import importlib
from pathlib import Path
names = [
    "torch", "torchvision", "yaml", "PIL", "psutil", "onnxruntime",
    "rapidocr_onnxruntime", "transformers", "cv2", "numpy",
]
lines = []
for name in names:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "ok")
        lines.append(f"{name} {version}")
        if name == "torch":
            lines.append(f"cuda {module.cuda.is_available()} device_count {module.cuda.device_count()}")
    except Exception as exc:
        lines.append(f"{name} ERROR {type(exc).__name__} {exc}")
Path("/content/package_probe_extended.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
