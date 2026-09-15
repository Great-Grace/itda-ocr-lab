import subprocess,sys
r=subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-cache-dir','--no-deps','ultralytics==8.4.150'],text=True,capture_output=True)
print('return',r.returncode); print((r.stderr or r.stdout)[-5000:])
try:
 import torch; print('torch',torch.__version__,'cuda_available',torch.cuda.is_available())
 from ultralytics import YOLO; print('ultralytics=ok')
except Exception as exc: print('import_error',type(exc).__name__,exc)
