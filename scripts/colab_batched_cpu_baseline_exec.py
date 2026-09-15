"""Batch CPU PP-OCRv5 detector + PP-OCRv6 recognizer; emits token cache."""
from __future__ import annotations
import csv, json, os, time
from dataclasses import asdict
from pathlib import Path
import sys
os.environ.setdefault('FLAGS_use_mkldnn','0')
ROOT=Path('/content/itda_lab'); DRIVE=Path('/content/drive/MyDrive/상품사진입니다')
sys.path.insert(0, str(ROOT / 'src'))
from PIL import Image
import numpy as np
from paddleocr import TextDetection, TextRecognition
from ocr_lab.contracts import OCRToken
from ocr_lab.modules.paddle_ocr import _result_payload, _first_present, _crop_polygon, _bbox_from_poly

OUT=Path('/content/batched_cpu_baseline_tokens.jsonl'); META=Path('/content/batched_cpu_baseline_meta.json')
ids=[line.strip() for line in (ROOT/'data/splits/val_ids.txt').read_text().splitlines() if line.strip()]
limit=int(os.environ.get('BATCHED_LIMIT','0') or 0)
if limit:
 ids=ids[:limit]
paths=[]
for image_id in ids:
 for ext in ('.jpg','.jpeg','.png','.webp'):
  p=DRIVE/f'{image_id}{ext}'
  if p.exists(): paths.append((image_id,p)); break
print('images',len(paths),flush=True)
det=TextDetection(model_name='PP-OCRv5_mobile_det',model_dir='/content/drive/MyDrive/상품사진입니다/weights/paddle/ppocrv5_mobile_det',device='cpu',limit_side_len=960,thresh=0.25,box_thresh=0.50,unclip_ratio=1.8,enable_mkldnn=False)
rec=TextRecognition(model_name='PP-OCRv6_medium_rec',model_dir='/content/PP-OCRv6_medium_rec',device='cpu')
tokens_by_id={image_id:[] for image_id,_ in paths}; det_s=rec_s=0.0; crop_meta=[]
for image_id,p in paths:
 # PP-OCR detection requires same-shaped tensors for a batch, while product
 # photos have heterogeneous resolutions. Keep detector semantics identical
 # to the production per-image path and batch only recognition crops below.
 t=time.perf_counter(); result_list=list(det.predict(str(p),batch_size=1)); det_s+=time.perf_counter()-t
 result=result_list[0] if result_list else {}
 payload=_result_payload(result); polys=_first_present(payload,'dt_polys','polys','boxes'); scores=_first_present(payload,'dt_scores','det_scores')
 arr=np.asarray(Image.open(p).convert('RGB'))
 for idx,poly in enumerate(polys):
  crop=_crop_polygon(arr,poly,'axis_aligned')
  if crop is not None: crop_meta.append((image_id,poly,float(scores[idx]) if idx<len(scores) else None,crop))
for off in range(0,len(crop_meta),32):
 batch=crop_meta[off:off+32]; t=time.perf_counter(); rec_results=list(rec.predict([item[3] for item in batch],batch_size=len(batch))); rec_s+=time.perf_counter()-t
 for item,result in zip(batch,rec_results):
  image_id,poly,det_score,_=item; payload=_result_payload(result); texts=_first_present(payload,'rec_texts','text','rec_text'); scores=_first_present(payload,'rec_scores','scores','rec_score'); text=texts[0] if isinstance(texts,(list,tuple)) and texts else texts; score=scores[0] if isinstance(scores,(list,tuple)) and scores else scores
  if text is None: continue
  bbox=_bbox_from_poly(poly); extras={'polygon':poly.tolist() if hasattr(poly,'tolist') else poly,'source':'batched_cpu'}
  if bbox: l,t,r,b=bbox; extras.update({'center_x':(l+r)/2,'center_y':(t+b)/2,'width':r-l,'height':b-t})
  tokens_by_id[image_id].append(asdict(OCRToken(text=str(text),confidence=float(score or 0),bbox=bbox,extras=extras,detection_confidence=det_score)))
with OUT.open('w',encoding='utf-8') as f:
 for image_id in tokens_by_id: f.write(json.dumps({'image_id':image_id,'tokens':tokens_by_id[image_id]},ensure_ascii=False)+'\n')
meta={'images':len(paths),'crops':len(crop_meta),'detection_seconds':det_s,'recognition_seconds':rec_s,'total_seconds':det_s+rec_s,'sec_per_image':(det_s+rec_s)/max(1,len(paths)),'images_per_second':len(paths)/max(det_s+rec_s,1e-9)}
META.write_text(json.dumps(meta,ensure_ascii=False,indent=2)); print(json.dumps(meta),flush=True)
