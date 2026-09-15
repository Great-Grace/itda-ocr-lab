import subprocess,sys,importlib
cmd=[sys.executable,'-m','pip','install','--quiet','--no-cache-dir','--no-deps','paddleocr==3.7.0']
r=subprocess.run(cmd,text=True,capture_output=True,timeout=900)
print('return',r.returncode)
print((r.stderr or r.stdout)[-6000:])
try:
 import paddleocr
 print('paddleocr',getattr(paddleocr,'__version__','ok'))
 from paddleocr import TextRecognition
 print('TextRecognition=ok')
except Exception as exc: print('import_error',type(exc).__name__,exc)
