
import subprocess, sys

cmd = [sys.executable, "-u", "/content/itda_ocr/scripts/colab_measured_runner.py"]
log_file = open("/content/runner.log", "w", encoding="utf-8")
p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
print(f"LAUNCHED_PID: {p.pid}")
