import sys
import paddle
print('python',sys.version)
print('paddle',paddle.__version__)
print('compiled_cuda',paddle.is_compiled_with_cuda())
print('device',paddle.device.get_device())
paddle.utils.run_check()
print('run_check=ok')
