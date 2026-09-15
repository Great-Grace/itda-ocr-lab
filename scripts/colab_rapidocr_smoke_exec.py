from rapidocr_onnxruntime import RapidOCR
from pathlib import Path
import time, json, os
images=sorted(Path('/content/drive/MyDrive/상품사진입니다').glob('*.jpg'))
print('images',len(images),images[0] if images else None)
try:
    t=time.perf_counter(); engine=RapidOCR(); print('engine_init_s',time.perf_counter()-t)
    result, elapsed=engine(str(images[0]))
    print('rows',len(result or []),'elapsed',elapsed)
    print((result or [])[:3])
    Path('/content/rapidocr_smoke.json').write_text(json.dumps({'rows':len(result or []),'elapsed':elapsed},ensure_ascii=False),encoding='utf-8')
except Exception as exc:
    print(type(exc).__name__,exc)
