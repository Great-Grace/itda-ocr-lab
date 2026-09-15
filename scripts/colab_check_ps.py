
import subprocess
out = subprocess.run(["ps", "aux"], capture_output=True, text=True).stdout
for l in out.splitlines():
    if "python" in l:
        print(l)
