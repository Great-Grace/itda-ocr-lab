import subprocess, sys, importlib
from pathlib import Path
lines=[]
cmd=[sys.executable,'-m','pip','install','--quiet','--no-cache-dir','--no-deps','nvidia-cuda-nvrtc-cu13']
r=subprocess.run(cmd,text=True,capture_output=True,timeout=900)
lines.append(f'install_return={r.returncode}')
if r.stdout: lines.append(r.stdout[-3000:])
if r.stderr: lines.append(r.stderr[-6000:])
try:
    p=importlib.import_module('paddle'); lines.append(f'paddle={p.__version__} cuda={p.is_compiled_with_cuda()} device={p.device.get_device()}'); p.utils.run_check(); lines.append('run_check=ok')
except Exception as exc: lines.append(f'import_error={type(exc).__name__}: {exc}')
Path('/content/paddle_nvrtc_probe.txt').write_text('\\n'.join(lines),encoding='utf-8')
print('\\n'.join(lines))
