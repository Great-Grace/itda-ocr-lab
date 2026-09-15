
from pathlib import Path
log_p = Path("/content/runner.log")
if log_p.exists():
    lines = log_p.read_text(encoding="utf-8", errors="ignore").splitlines()
    print(f"=== runner.log (total {len(lines)} lines) ===")
    for l in lines[-25:]:
        print(l)
else:
    print("runner.log does not exist yet")
