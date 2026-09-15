import torch
print({'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'cuda_version': torch.version.cuda, 'devices': torch.cuda.device_count()})
