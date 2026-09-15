import subprocess, sys, importlib
from pathlib import Path
commands = [
    [sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", "paddlepaddle-gpu==3.2.2"],
    [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "paddleocr==3.3.0"],
]
lines=[]
for cmd in commands:
    r=subprocess.run(cmd,text=True,capture_output=True)
    lines.append(f"{' '.join(cmd[3:])} return={r.returncode}")
    if r.returncode:
        lines.append((r.stderr or r.stdout)[-5000:])
        break
try:
    paddle=importlib.import_module('paddle')
    lines.append(f"paddle={paddle.__version__} compiled={paddle.is_compiled_with_cuda()} devices={paddle.device.get_device()}")
except Exception as exc:
    lines.append(f"paddle_import_error={type(exc).__name__}: {exc}")
try:
    pocr=importlib.import_module('paddleocr')
    lines.append(f"paddleocr={getattr(pocr,'__version__','ok')}")
except Exception as exc:
    lines.append(f"paddleocr_import_error={type(exc).__name__}: {exc}")
Path('/content/paddle_probe.txt').write_text('\n'.join(lines),encoding='utf-8')
print('\n'.join(lines))
