import subprocess,sys
cmd=[sys.executable,'-m','pip','install','--quiet','--no-cache-dir','paddlex[ocr-core]>=3.7.0,<3.8.0']
r=subprocess.run(cmd,text=True,capture_output=True,timeout=1200)
print('return',r.returncode)
print((r.stderr or r.stdout)[-10000:])
try:
 import paddlex
 print('paddlex',getattr(paddlex,'__version__','ok'))
 from paddleocr import TextRecognition
 print('TextRecognition=ok')
except Exception as exc: print('import_error',type(exc).__name__,exc)
