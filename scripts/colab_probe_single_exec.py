from pathlib import Path
import importlib
log = Path('/content/package_probe.txt')
log.write_text('start\n', encoding='utf-8')
for name in ['torch', 'numpy', 'cv2', 'paddle', 'paddleocr', 'ultralytics']:
    try:
        module = importlib.import_module(name)
        line = f'{name}: {getattr(module, "__version__", "ok")}\n'
    except Exception as exc:
        line = f'{name}: ERROR {type(exc).__name__}: {exc}\n'
    with log.open('a', encoding='utf-8') as handle:
        handle.write(line)
print(log.read_text(encoding='utf-8'))
