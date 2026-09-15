from pathlib import Path
import sys
dataset = Path('/content/drive/MyDrive/상품사진입니다')
print('dataset', dataset, dataset.is_dir(), flush=True)
print('files', sum(1 for p in dataset.iterdir() if p.is_file()) if dataset.is_dir() else -1, flush=True)
archive = dataset / 'weights/ppocrv6_medium_rec.tar.gz'
print('archive', archive, archive.is_file(), archive.stat().st_size if archive.is_file() else -1, flush=True)
print('python', sys.executable, flush=True)
try:
    import paddle, paddleocr
    print('paddle', paddle.__version__, 'paddleocr', getattr(paddleocr, '__version__', None), flush=True)
except Exception as exc:
    print('import_error', repr(exc), flush=True)
