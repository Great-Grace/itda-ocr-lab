import json,time
from pathlib import Path
import torch
from ultralytics import YOLO
paths=sorted(Path('/content/itda_t4_val_images').glob('*.jpg'))[:32]
model=YOLO('/content/expiry_binary_yolov8n_1280_best.pt')
t=time.perf_counter(); res=model.predict(source=[str(p) for p in paths],imgsz=960,conf=0.2,device=0,batch=32,verbose=False,stream=False); torch.cuda.synchronize(); dt=time.perf_counter()-t
print(json.dumps({'images':len(res),'seconds':dt,'sec_per_image':dt/len(res),'boxes':sum(len(r.boxes) for r in res),'gpu_mem':torch.cuda.max_memory_allocated()/1024/1024}))
