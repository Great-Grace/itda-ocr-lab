from .date_normalizer import DateNormalizer
from .mock_ocr import MockOCRBackend
from .paddle_ocr import PaddleOCRBackend
from .preprocessor import (
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
}

OCR_BACKENDS = {
    "mock": MockOCRBackend,
    "paddle_mobile": PaddleOCRBackend,
}

SELECTORS = {"keyword_regex": KeywordRegexSelector}
NORMALIZERS = {"date_ko_v1": DateNormalizer}
