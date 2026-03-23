
import torch.nn as nn
import torch.nn.functional as F
import torch
#nn.Module is a base class for all neural networks

class SimpleCNN(nn.Module):
    def __init__(self,num_classes=10):
        super(SimpleCNN,self).__init__()# calling parent class
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.block2= nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.block3= nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fullyconnedtedLayer= nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4,256),
            nn.ReLU(),
            nn.Linear(256,num_classes),
        )
         
        
    
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.fullyconnedtedLayer(x)
        return x
    


if __name__== "__main__":
    # device configuration for GPU usage
    device= torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model= SimpleCNN(num_classes=10).to(device)


    