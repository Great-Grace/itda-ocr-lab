import inspect
import paddleocr
from paddleocr import TextRecognition
print(paddleocr.__file__)
print(inspect.getsource(TextRecognition)[:12000])
root = __import__('pathlib').Path(paddleocr.__file__).parent
for path in root.rglob('*'):
    if path.is_file() and path.suffix in {'.py', '.json', '.yaml', '.yml'}:
        try:
            text = path.read_text(errors='ignore')
        except Exception:
            continue
        if 'korean_PP' in text or 'PP-OCRv6' in text:
            print('MODEL_REF', path)
            for line in text.splitlines():
                if 'korean_PP' in line or 'PP-OCRv6' in line:
                    print(line[:300])
