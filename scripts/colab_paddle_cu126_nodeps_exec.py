import subprocess,sys,importlib
from pathlib import Path
lines=[]
def run(cmd,timeout=1200):
    r=subprocess.run(cmd,text=True,capture_output=True,timeout=timeout)
    lines.append('$ '+' '.join(cmd)+f' return={r.returncode}')
    if r.stdout: lines.append(r.stdout[-3000:])
    if r.stderr: lines.append(r.stderr[-7000:])
    return r.returncode
run([sys.executable,'-m','pip','uninstall','-y','paddlepaddle-gpu'])
run([sys.executable,'-m','pip','install','--quiet','--no-cache-dir','--no-deps','paddlepaddle-gpu==3.3.0','-i','https://www.paddlepaddle.org.cn/packages/stable/cu126/'])
try:
    p=importlib.import_module('paddle'); lines.append(f'paddle={p.__version__} cuda={p.is_compiled_with_cuda()} device={p.device.get_device()}'); p.utils.run_check(); lines.append('run_check=ok')
except Exception as exc: lines.append(f'import_error={type(exc).__name__}: {exc}')
Path('/content/paddle_cu126_probe.txt').write_text('\\n'.join(lines),encoding='utf-8')
print('\\n'.join(lines))
