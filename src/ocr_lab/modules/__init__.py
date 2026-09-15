from .amsc_cascade_ocr import AMSC_CascadeOCRBackend
from .cascade_ocr import CascadeOCRBackend
from .date_normalizer import DateNormalizer
from .line_reconstruction import reconstruct_lines
from .mock_ocr import MockOCRBackend
from .paddle_ocr import PaddleOCRBackend, PaddleOCRSplitBackend
from .rapid_ocr import RapidOCRBackend
from .union_ocr import PaddleYoloUnionBackend
from .union_spatial_selector import UnionSpatialSelector
from .preprocessor import (
    AdaptiveContrastPreprocessor,
    LocalClahePreprocessor,
    GrayscaleContrastPreprocessor,
    PassthroughPreprocessor,
    ResizePreprocessor,
)
from .regex_selector import KeywordRegexSelector


PREPROCESSORS = {
    "none": PassthroughPreprocessor,
    "passthrough": PassthroughPreprocessor,
    "resize": ResizePreprocessor,
    "grayscale_contrast": GrayscaleContrastPreprocessor,
    "adaptive_contrast": AdaptiveContrastPreprocessor,
    "clahe": AdaptiveContrastPreprocessor,
    "clahe_local": LocalClahePreprocessor,
}

OCR_BACKENDS = {
    "mock": MockOCRBackend,
    "paddle_mobile": PaddleOCRBackend,
    "paddle_mobile_split": PaddleOCRSplitBackend,
    "rapid_onnx": RapidOCRBackend,
    "cascade": CascadeOCRBackend,
    "amsc_cascade": AMSC_CascadeOCRBackend,
    "paddle_yolo_union": PaddleYoloUnionBackend,
}

SELECTORS = {"keyword_regex": KeywordRegexSelector, "union_spatial": UnionSpatialSelector}
NORMALIZERS = {"date_ko_v1": DateNormalizer}
