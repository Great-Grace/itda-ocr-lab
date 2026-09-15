import os, sys, traceback
import importlib
from pathlib import Path
root = Path('/content/itda_ocr_lab')
sys.path.insert(0, str(root / 'src'))
importlib.invalidate_caches()
print('cwd', os.getcwd(), 'path', sys.path[:4], 'src', list((root/'src').iterdir())[:4])
print('exists', (root/'src'/'ocr_lab').exists(), (root/'src'/'ocr_lab'/'__init__.py').exists(), os.listdir(root/'src'/'ocr_lab'))
import importlib.util
print('spec', importlib.util.find_spec('ocr_lab'))
try:
    import ocr_lab
    print('imported', ocr_lab, getattr(ocr_lab, '__file__', None))
except Exception:
    traceback.print_exc()
