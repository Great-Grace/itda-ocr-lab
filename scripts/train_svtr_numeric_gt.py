"""Train a numeric-only SVTR CTC recognizer on annotated date crops."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ocr_lab.modules.attention_svtr import SVTRTinyCTC, ctc_greedy_decode


CHARSET = "0123456789"


class NumericCropDataset(Dataset):
    def __init__(self, root, rows, augment=False): self.root,self.rows,self.augment=Path(root),rows,augment
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]; image=Image.open(self.root/row['image']).convert('L')
        if self.augment:
            if random.random()<.4: image=image.filter(ImageFilter.GaussianBlur(random.uniform(.1,.6)))
            if random.random()<.3:
                arr=np.asarray(image,dtype=np.float32)+np.random.normal(0,4,np.asarray(image).shape)
                image=Image.fromarray(np.clip(arr,0,255).astype('uint8'))
        arr=np.asarray(image,dtype=np.float32)/255.; text=''.join(c for c in row['label'] if c.isdigit())
        return torch.from_numpy(arr).unsqueeze(0),text


def collate(batch):
    images,texts=zip(*batch); targets=torch.tensor([int(c)+1 for text in texts for c in text],dtype=torch.long); lengths=torch.tensor([len(text) for text in texts],dtype=torch.long); return torch.stack(images),targets,lengths,list(texts)


def edit(a,b):
    p=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        c=[i]
        for j,y in enumerate(b,1): c.append(min(c[-1]+1,p[j]+1,p[j-1]+(x!=y)))
        p=c
    return p[-1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--train-crops',required=True);p.add_argument('--val-crops',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=30);p.add_argument('--device',default='cpu');args=p.parse_args()
    train_rows=[r for r in csv.DictReader(open(Path(args.train_crops)/'labels.csv')) if 'None' not in r['label']]
    val_rows=[r for r in csv.DictReader(open(Path(args.val_crops)/'labels.csv')) if 'None' not in r['label']]
    device=torch.device(args.device if args.device=='cpu' or torch.cuda.is_available() else 'cpu')
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=SVTRTinyCTC(len(CHARSET)).to(device)
    base=checkpoint['state_dict']; compatible={k:v for k,v in base.items() if k in model.state_dict() and model.state_dict()[k].shape==v.shape}; model.load_state_dict(compatible,strict=False)
    train=DataLoader(NumericCropDataset(args.train_crops,train_rows,True),batch_size=16,shuffle=True,num_workers=0,collate_fn=collate)
    val=DataLoader(NumericCropDataset(args.val_crops,val_rows),batch_size=16,shuffle=False,num_workers=0,collate_fn=collate)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=1e-4,eps=1e-6);loss_fn=nn.CTCLoss(blank=0,zero_infinity=True);history=[]
    for epoch in range(1,args.epochs+1):
        model.train();loss_total=0
        for images,targets,lengths,_ in train:
            images,targets,lengths=images.to(device),targets.to(device),lengths.to(device); logits=model(images); input_lengths=torch.full((images.size(0),),logits.size(0),dtype=torch.long,device=device); loss=loss_fn(logits,targets,input_lengths,lengths);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step();loss_total+=float(loss.item())
        model.eval();correct=total=dist=chars=0
        with torch.no_grad():
            for images,_,_,texts in val:
                pred=ctc_greedy_decode(model(images.to(device)),CHARSET); correct+=sum(a==b for a,b in zip(pred,texts)); total+=len(texts); dist+=sum(edit(a,b) for a,b in zip(pred,texts)); chars+=sum(max(1,len(b)) for b in texts)
        row={'epoch':epoch,'train_loss':loss_total/max(1,len(train)),'val_exact':correct/max(1,total),'val_cer':dist/max(1,chars)};history.append(row);print(json.dumps(row),flush=True)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);torch.save({'state_dict':model.state_dict(),'charset':CHARSET,'model':'SVTRTinyCTC_NUMERIC_GT'},out/'checkpoint.pt');(out/'history.json').write_text(json.dumps(history,indent=2))

if __name__=='__main__': main()
