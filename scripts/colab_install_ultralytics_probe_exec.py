import subprocess, sys
from pathlib import Path
cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "ultralytics==8.4.150"]
result = subprocess.run(cmd, text=True, capture_output=True)
lines = [f"pip_returncode={result.returncode}"]
if result.stdout:
    lines.append(result.stdout[-2000:])
if result.stderr:
    lines.append(result.stderr[-4000:])
try:
    import ultralytics
    lines.append(f"ultralytics={getattr(ultralytics, '__version__', 'ok')}")
except Exception as exc:
    lines.append(f"import_error={type(exc).__name__}: {exc}")
Path("/content/ultralytics_probe.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
