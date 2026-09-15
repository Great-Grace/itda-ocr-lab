import subprocess, sys, importlib
from pathlib import Path
cmd=[sys.executable,'-m','pip','install','--quiet','--no-cache-dir','paddlepaddle-gpu==3.3.0','-i','https://www.paddlepaddle.org.cn/packages/stable/cu130/']
r=subprocess.run(cmd,text=True,capture_output=True,timeout=1200)
lines=[f'python={sys.version}',f'return={r.returncode}']
if r.stdout: lines.append(r.stdout[-5000:])
if r.stderr: lines.append(r.stderr[-8000:])
try:
    paddle=importlib.import_module('paddle')
    lines.append(f'paddle={paddle.__version__} compiled_cuda={paddle.is_compiled_with_cuda()} device={paddle.device.get_device()}')
    lines.append(str(paddle.utils.run_check()))
except Exception as exc: lines.append(f'import_error={type(exc).__name__}: {exc}')
Path('/content/paddle_cu130_probe.txt').write_text('\\n'.join(lines),encoding='utf-8')
print('\\n'.join(lines))
