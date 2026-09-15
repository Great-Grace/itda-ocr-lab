import sys, subprocess, os
print(sys.executable, sys.path)
print(subprocess.run([sys.executable,'-c','import sys; print(sys.executable); import paddleocr; print(paddleocr.__file__)'],capture_output=True,text=True).stdout)
print(subprocess.run(['/usr/local/bin/python3','-c','import paddleocr; print(paddleocr.__file__)'],capture_output=True,text=True).stdout)
print(subprocess.run([sys.executable,'-m','pip','show','paddleocr'],capture_output=True,text=True).stdout)
