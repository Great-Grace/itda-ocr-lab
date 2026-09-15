import sys
print('python',sys.version)
try:
 import paddle, paddlex
 print('paddle',paddle.__version__,paddle.is_compiled_with_cuda(),paddle.device.get_device())
 print('paddlex',getattr(paddlex,'__version__','ok'))
 from paddleocr import TextRecognition
 print('TextRecognition=ok')
except Exception as exc:
 print('ERROR',type(exc).__name__,exc)
