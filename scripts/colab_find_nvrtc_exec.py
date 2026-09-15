from pathlib import Path
for root in ('/usr/local','/usr','/opt'):
    p=Path(root)
    if p.exists():
        for x in p.rglob('libnvrtc.so*'):
            print(x)
