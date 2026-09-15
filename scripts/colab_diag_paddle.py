import paddle, paddleocr, paddlex
print('paddle', paddle.__version__, 'paddleocr', getattr(paddleocr, '__version__', 'unknown'), 'paddlex', getattr(paddlex, '__version__', 'unknown'))
print('device', paddle.get_device())
