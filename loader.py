import os 
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


classes=["Background","Messi","Ronaldo","Neymar"]

class Dataset(Dataset):
    
    def __init__(self,root_dir,annotations_file,transform=None):
        self.root_dir = root_dir
        self.df=pd.read_csv(annotations_file)
        self.transform = transform
        
        self.imgs=self.df['filename']
        self.classification=self.df['class']
        self.xmin=self.df['xmin']
        self.ymin=self.df['ymin']
        self.xmax=self.df['xmax']
        self.ymax=self.df['ymax']
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self,idx):
        img_id=self.imgs[idx]
        img=Image.open(os.path.join(self.root_dir,img_id)).convert('RGB')

        boxes=[[self.xmin[idx],self.ymin[idx],self.xmax[idx],self.ymax[idx]]]
        boxes=torch.as_tensor(boxes,dtype=torch.float32)
        labels=torch.tensor([classes.index(self.classification[idx])],dtype=torch.int64)
        image_id=torch.tensor([idx])
        area=(boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((1,), dtype=torch.int64)
        
        target={}
        target["boxes"]=boxes
        target["labels"]=labels
        target["image_id"]=image_id
        target["area"]=area
        target["iscrowd"]=iscrowd
        
        if self.transform is not None:
            img=self.transform(img)
        
        return img,target
    
    
if __name__ == '__main__':
    transform=transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.RandomHorizontalFlip(0.5)
    ])

    train_dataset=Dataset(root_dir='train',annotations_file='train/_annotations.csv',transform=transform)
    print(train_dataset[0])
