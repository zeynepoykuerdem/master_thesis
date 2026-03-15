import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
class ZhangExperiment(Dataset):
    def __init__(self, dataset, experiment='true'):
        """
        experiment:
        'true'-> original data
        'random_labels'-> shuffled labels
        'random pixels'-> random pixels

        """

        self.dataset= dataset
        self.experiment= experiment
        self.n= len(dataset)

        if experiment== 'random_labels':
            self.random_labels=np.random.randint(0,10,self.n)

    def __len__(self):
        return self.n
    
    def __getitem__(self,idx):
        image,label= self.dataset[idx]

        if self.experiment=='true':
            return image,label
        
        elif self.experiment=='random_labels':
            return image,self.random_labels[idx]

        elif self.experiment=='random_pixels':
            random_image=torch.rand_like(image)
            return random_image,label
        elif self.experiment== 'shuffled_pixels':
            flat= image.view(-1)
            perm= torch.randperm(flat.size(0))
            shuffled=flat[perm].view(image.size())
            return shuffled,label





