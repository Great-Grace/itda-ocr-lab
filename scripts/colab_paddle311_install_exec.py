from pathlib import Path
import subprocess
py=Path('/content/paddle311/bin/python')
lines=[]
def run(cmd, timeout=1200):
    r=subprocess.run(cmd,text=True,capture_output=True,timeout=timeout)
    lines.append('$ '+' '.join(map(str,cmd))+f' return={r.returncode}')
    if r.stdout: lines.append(r.stdout[-3000:])
    if r.stderr: lines.append(r.stderr[-6000:])
    return r.returncode
run([str(py),'-m','ensurepip','--upgrade'])
run([str(py),'-m','pip','install','--quiet','--no-cache-dir','--index-url','https://www.paddlepaddle.org.cn/packages/stable/cu126/','paddlepaddle-gpu==3.2.2'])
run([str(py),'-c','import paddle; print(paddle.__version__, paddle.is_compiled_with_cuda(), paddle.device.get_device())'])
Path('/content/paddle311_install_probe.txt').write_text('\\n'.join(lines),encoding='utf-8')
print('\\n'.join(lines))
