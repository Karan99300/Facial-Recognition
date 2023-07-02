import torch
from tqdm import tqdm

def train_step(model,train_loader,optimizer,device):
    model.train()
    epoch_loss=[]
    pbar=tqdm(train_loader)
    for i,(images,targets) in enumerate(pbar):    
        images=list(image.to(device) for image in images)
        targets=[{k: v.to(device) for k,v in t.items()} for t in targets]
        
        loss_dict=model(images,targets)    
        loss=sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        loss.backward()
        epoch_loss.append(loss.item())
        optimizer.step()  
        
    return (sum(epoch_loss)/len(epoch_loss))

