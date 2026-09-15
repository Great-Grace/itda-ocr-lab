"""Train SVTR-Tiny CTC on GT date crops and evaluate a held-out crop set."""
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


class CropDataset(Dataset):
    def __init__(self, root, rows, charset, augment=False): self.root,self.rows,self.charset,self.augment=Path(root),rows,charset,augment
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]; image=Image.open(self.root/row['image']).convert('L')
        if self.augment and random.random()<.5: image=image.filter(ImageFilter.GaussianBlur(random.uniform(.1,.5)))
        arr=np.asarray(image,dtype=np.float32)/255.; return torch.from_numpy(arr).unsqueeze(0),row['raw_text']


def collate(batch,charset):
    images,labels=zip(*batch); targets=torch.tensor([charset.index(c)+1 for label in labels for c in label],dtype=torch.long); lengths=torch.tensor([len(label) for label in labels],dtype=torch.long); return torch.stack(images),targets,lengths,list(labels)


def edit(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1): cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--train-crops',required=True);p.add_argument('--val-crops',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=25);args=p.parse_args()
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False);charset=checkpoint['charset']
    train_rows=[r for r in csv.DictReader(open(Path(args.train_crops)/'labels.csv')) if set(r['raw_text'])<=set(charset)]
    val_rows=[r for r in csv.DictReader(open(Path(args.val_crops)/'labels.csv')) if set(r['raw_text'])<=set(charset)]
    model=SVTRTinyCTC(len(charset));model.load_state_dict(checkpoint['state_dict']);model.train()
    train_loader=DataLoader(CropDataset(args.train_crops,train_rows,charset,True),batch_size=16,shuffle=True,num_workers=0,collate_fn=lambda b:collate(b,charset))
    val_loader=DataLoader(CropDataset(args.val_crops,val_rows,charset),batch_size=16,shuffle=False,num_workers=0,collate_fn=lambda b:collate(b,charset))
    opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=1e-4,eps=1e-6);loss_fn=nn.CTCLoss(blank=0,zero_infinity=True);history=[]
    for epoch in range(1,args.epochs+1):
        model.train();total_loss=0
        for images,targets,lengths,_ in train_loader:
            logits=model(images);input_lengths=torch.full((images.size(0),),logits.size(0),dtype=torch.long);loss=loss_fn(logits,targets,input_lengths,lengths);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step();total_loss+=float(loss.item())
        model.eval();correct=total=dist=chars=0
        with torch.no_grad():
            for images,_,_,labels in val_loader:
                preds=ctc_greedy_decode(model(images),charset);correct+=sum(a==b for a,b in zip(preds,labels));total+=len(labels);dist+=sum(edit(a,b) for a,b in zip(preds,labels));chars+=sum(max(1,len(b)) for b in labels)
        row={'epoch':epoch,'train_loss':total_loss/max(1,len(train_loader)),'val_exact':correct/max(1,total),'val_cer':dist/max(1,chars)};history.append(row);print(json.dumps(row),flush=True)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);torch.save({'state_dict':model.state_dict(),'charset':charset,'model':'SVTRTinyCTC_GT_CROPS'},out/'checkpoint.pt');(out/'history.json').write_text(json.dumps(history,indent=2))

if __name__=='__main__':main()
