import subprocess, sys, importlib
from pathlib import Path
cmd=[sys.executable,'-m','pip','install','--quiet','--no-cache-dir','onnxruntime-gpu','rapidocr_onnxruntime']
r=subprocess.run(cmd,text=True,capture_output=True)
lines=[f'return={r.returncode}',(r.stderr or r.stdout)[-6000:]]
for name in ['onnxruntime','rapidocr_onnxruntime']:
    try:
        m=importlib.import_module(name); lines.append(f'{name}={getattr(m,"__version__","ok")}')
    except Exception as exc: lines.append(f'{name}_error={type(exc).__name__}: {exc}')
Path('/content/rapidocr_probe.txt').write_text('\\n'.join(lines),encoding='utf-8')
print('\\n'.join(lines))
