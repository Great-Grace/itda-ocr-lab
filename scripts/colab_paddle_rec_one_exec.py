import time, json
from pathlib import Path
import numpy as np
from PIL import Image
from paddleocr import TextRecognition
model_dir='/content/PP-OCRv6_medium_rec'
path=sorted(Path('/content/drive/MyDrive/상품사진입니다').glob('*.jpg'))[0]
print('model_dir',model_dir,'files',sorted(p.name for p in Path(model_dir).glob('*')))
print('image',path)
t=time.perf_counter(); rec=TextRecognition(model_name='PP-OCRv6_medium_rec',model_dir=model_dir,device='gpu:0'); print('init_s',time.perf_counter()-t)
arr=np.asarray(Image.open(path).convert('RGB'))
t=time.perf_counter(); out=rec.predict([arr],batch_size=1); print('predict_s',time.perf_counter()-t); print(out)
