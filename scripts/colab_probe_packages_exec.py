import importlib
from pathlib import Path
lines = []
for name in ["torch", "paddle", "paddleocr", "ultralytics", "cv2", "numpy"]:
    try:
        module = importlib.import_module(name)
        lines.append(f"{name} {getattr(module, '__version__', 'ok')}")
        if name == "torch": lines.append(f"cuda {module.cuda.is_available()}")
    except Exception as exc:
        lines.append(f"{name} ERROR {type(exc).__name__} {exc}")
Path("/content/package_probe.txt").write_text("\\n".join(lines), encoding="utf-8")
print("\\n".join(lines))
