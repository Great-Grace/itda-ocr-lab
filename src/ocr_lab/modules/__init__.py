from .date_normalizer import DateNormalizer
from .mock_ocr import MockOCRBackend
from .paddle_ocr import PaddleOCRBackend
from .regex_selector import KeywordRegexSelector


OCR_BACKENDS = {
    "mock": MockOCRBackend,
    "paddle_mobile": PaddleOCRBackend,
}

SELECTORS = {"keyword_regex": KeywordRegexSelector}
NORMALIZERS = {"date_ko_v1": DateNormalizer}
