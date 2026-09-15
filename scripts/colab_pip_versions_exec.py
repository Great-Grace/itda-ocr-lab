import subprocess,sys
r=subprocess.run([sys.executable,'-m','pip','index','versions','nvidia-cuda-nvrtc-cu13'],text=True,capture_output=True)
print(r.stdout); print(r.stderr)
