import base64
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "colab_fixed_cli.py"

labeler_code = (ROOT / "scripts" / "gemini_vlm_labeler.py").read_text(encoding="utf-8")
labeler_b64 = base64.b64encode(labeler_code.encode("utf-8")).decode("ascii")

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    env_file = ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                api_key = line.split("=", 1)[1].strip()

local_csv = ROOT / "runs" / "gemini_vlm_labels" / "gemini_labels.csv"
csv_b64 = ""
if local_csv.is_file():
    csv_b64 = base64.b64encode(local_csv.read_bytes()).decode("ascii")

remote_script = f"""
import base64, os, subprocess, sys, csv, shutil
from pathlib import Path

# 0. Terminate any previous labeler processes
subprocess.run(["pkill", "-9", "-f", "gemini_vlm_labeler.py"])
print("Terminated any previous gemini_vlm_labeler processes.")

# 1. Deploy latest labeler script
target = Path('/content/gemini_vlm_labeler.py')
target.write_text(base64.b64decode('{labeler_b64}').decode('utf-8'), encoding='utf-8')
print('Deployed latest labeler script with gemini-3.5-flash-lite!')

# 2. Restore and clean up CSV
out_dir = Path('/content/runs/gemini_vlm_labels')
out_dir.mkdir(parents=True, exist_ok=True)
csv_path = out_dir / 'gemini_labels.csv'
drive_csv = Path('/content/drive/MyDrive/gemini_labels.csv')

if drive_csv.is_file() and drive_csv.stat().st_size > 100:
    shutil.copyfile(drive_csv, csv_path)
    print(f'Restored {{len(csv_path.read_text().splitlines())}} rows from Google Drive!')
elif not csv_path.is_file() and '{csv_b64}':
    csv_path.write_bytes(base64.b64decode('{csv_b64}'))
    print(f'Restored {{len(csv_path.read_text().splitlines())}} rows from local backup!')

clean_count = 0
if csv_path.is_file():
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        clean_rows = [r for r in reader if r.get('reason') != 'API fail' and r.get('confidence') != 'none']
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('filename,date,confidence,text_found,reason\\n')
        for r in clean_rows:
            f.write(f'\"{{r[\"filename\"]}}\",\"{{r[\"date\"]}}\",\"{{r[\"confidence\"]}}\",\"{{r[\"text_found\"]}}\",\"{{r[\"reason\"]}}\"\\n')
    clean_count = len(clean_rows)
    print(f'Cleaned CSV: preserved {{clean_count}} verified clean labels!')

# 3. Launch detached background process
env = os.environ.copy()
env['GEMINI_API_KEY'] = '{api_key}'
env['PYTHONUNBUFFERED'] = '1'

log_file = open('/content/runs/gemini_vlm_labels/daemon.log', 'w')
cmd = [
    sys.executable, '-u', '/content/gemini_vlm_labeler.py',
    '--input-dir', '/content/drive/MyDrive/상품사진입니다',
    '--output-dir', '/content/runs/gemini_vlm_labels',
    '--batch-size', '4',
    '--delay', '4.2'
]
proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
print(f'SUCCESS: Daemon started with PID {{proc.pid}}!')
"""

tmp_file = Path("/tmp/remote_daemon_starter.py")
tmp_file.write_text(remote_script, encoding="utf-8")

session_name = "itda-labeler-3"
cmd = [sys.executable, str(CLI), "exec", "-s", session_name, "-f", str(tmp_file), "--timeout", "60"]
res = subprocess.run(cmd, capture_output=True, text=True)
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)


