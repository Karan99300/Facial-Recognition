import torch
from loader import Dataset
from tqdm import tqdm
from model import FasterRCNN
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import utils
import matplotlib.pyplot as plt
from engine import train_step

classes=["Background","Messi","Ronaldo","Neymar"]
num_classes=len(classes)

device=torch.device('cuda') 

def train():
    
    transform=transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
    ])
    
    train_dataset=Dataset(root_dir='train',annotations_file='train/_annotations.csv',transform=transform)
    
    train_loader=DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=utils.collate_fn
    )

    
    
    train_loss_vals=[]
    
    num_epochs=100
    model=FasterRCNN(num_classes=num_classes)
    model.to(device)
    
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=optim.SGD(params,lr=0.001,weight_decay=0.0005,momentum=0.9)
        
    for epoch in tqdm(range(num_epochs)):
        train_loss=train_step(
            model=model,train_loader=train_loader,optimizer=optimizer,device=device
        )
        train_loss_vals.append(train_loss)
        
        print('\n')
        print(f"Epoch:{epoch+1} , Training Loss:{train_loss}")
        
    
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.plot(train_loss_vals)
    plt.savefig("TrainingLoss.jpg")
    
    
    torch.save(model.state_dict(),'faster_rcnn.pth')
    
            
        
if __name__ == '__main__': 
    train()