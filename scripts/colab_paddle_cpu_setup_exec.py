import subprocess,sys
from pathlib import Path
lines=[]
def run(cmd,timeout=1200):
 r=subprocess.run(cmd,text=True,capture_output=True,timeout=timeout); lines.append('$ '+' '.join(cmd)+f' return={r.returncode}'); lines.append((r.stderr or r.stdout)[-7000:]); return r.returncode
run([sys.executable,'-m','pip','install','--quiet','--no-cache-dir','--no-deps','paddlepaddle==3.3.0','-i','https://www.paddlepaddle.org.cn/packages/stable/cpu/'])
run([sys.executable,'-m','pip','install','--quiet','--no-cache-dir','paddleocr==3.7.0','paddlex[ocr-core]>=3.7.0,<3.8.0'])
try:
 import paddle; lines.append(f'paddle={paddle.__version__} cuda={paddle.is_compiled_with_cuda()} device={paddle.device.get_device()}')
 from paddleocr import TextRecognition; lines.append('TextRecognition=ok')
except Exception as exc: lines.append(f'import_error={type(exc).__name__}: {exc}')
Path('/content/paddle_cpu_setup.txt').write_text('\\n'.join(lines),encoding='utf-8'); print('\\n'.join(lines))
