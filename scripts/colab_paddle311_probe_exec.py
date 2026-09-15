"""Probe a Python 3.11 venv for Paddle GPU on the Python 3.13 Colab image."""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path('/content/paddle311')
UV = Path('/content/uv')
lines = []

def run(cmd: list[str], timeout: int = 900) -> int:
    lines.append('$ ' + ' '.join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    lines.append(f'return={result.returncode}')
    if result.stdout:
        lines.append(result.stdout[-3000:])
    if result.stderr:
        lines.append(result.stderr[-6000:])
    return result.returncode

if not UV.exists():
    run(['python', '-m', 'pip', 'install', '--quiet', '--no-cache-dir', 'uv'])
if not (ROOT / 'bin/python').exists():
    run(['uv', 'python', 'install', '3.11'])
    run(['uv', 'venv', '--python', '3.11', str(ROOT)])
py = str(ROOT / 'bin/python')
run([py, '-m', 'pip', 'install', '--quiet', '--no-cache-dir', '--index-url', 'https://www.paddlepaddle.org.cn/packages/stable/cu126/', 'paddlepaddle-gpu==3.2.2'], timeout=1200)
run([py, '-c', 'import paddle; print(paddle.__version__, paddle.is_compiled_with_cuda(), paddle.device.get_device())'])
Path('/content/paddle311_probe.txt').write_text('\\n'.join(lines), encoding='utf-8')
print('\\n'.join(lines))
