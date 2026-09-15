import rapidocr_onnxruntime, inspect
from pathlib import Path
import rapidocr_onnxruntime.ch_ppocr_det as det
print('pkg', rapidocr_onnxruntime.__file__)
from rapidocr_onnxruntime import RapidOCR
print('det sig',inspect.signature(det.TextDetector))
print(inspect.getsource(det.TextDetector.__init__))
