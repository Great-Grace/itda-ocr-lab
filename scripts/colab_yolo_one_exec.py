import json, time
from pathlib import Path
import torch
print('start')
print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
from ultralytics import YOLO
print('imported')
model=YOLO('/content/expiry_binary_yolov8n_1280_best.pt')
print('model_loaded')
path=sorted(Path('/content/itda_t4_val_images').glob('*.jpg'))[0]
print('path',path)
t=time.perf_counter(); res=model.predict(source=str(path),imgsz=960,conf=0.2,device=0,verbose=False); torch.cuda.synchronize(); dt=time.perf_counter()-t
print('done',dt,len(res),len(res[0].boxes))
print(json.dumps({'seconds':dt,'boxes':len(res[0].boxes)}))
