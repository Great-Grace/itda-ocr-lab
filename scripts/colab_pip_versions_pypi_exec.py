import subprocess,sys
r=subprocess.run([sys.executable,'-m','pip','index','versions','nvidia-cuda-nvrtc-cu13','--index-url','https://pypi.org/simple'],text=True,capture_output=True)
print(r.stdout); print(r.stderr)
