import inspect
from paddleocr import TextDetection
print(inspect.signature(TextDetection))
print(inspect.signature(TextDetection.predict))
