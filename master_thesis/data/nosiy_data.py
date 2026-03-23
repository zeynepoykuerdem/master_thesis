import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
class ZhangExperiment(Dataset):
    def __init__(self, dataset,experiment='true_labels', noise_ratio=1.0):
        """
        experiment:
        'true_labels'-> original data
        'random_labels'-> shuffled labels
        'random_pixels'-> random pixels
        """

        self.dataset= dataset
        self.experiment= experiment
        self.n= len(dataset)
        
        sample_image, _ = dataset[0]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        c,h,w= sample_image.shape

        if experiment== 'random_pixels':
            # Generate a gaussian random noise with mean 0 and std 1, and shuffle the pixels of each image using the same random permutation
            self.static_noise= torch.randn(self.n, c, h, w)

        elif experiment== 'random_labels':
            n_noisy= int(self.n*noise_ratio)
            noisy_idx=torch.randperm(self.n)[:n_noisy]
            random_labels= torch.randint(0,10,(n_noisy,))
            self.label_map ={
                idx: random_labels[i].item()
                for i, idx in enumerate(noisy_idx.tolist())
            }

    def __len__(self):
        return self.n
    
    def __getitem__(self,idx):
        image,label= self.dataset[idx]


        if self.experiment=='true_labels':
            return image,label
        
        elif self.experiment=='random_labels':
            return image,self.label_map.get(idx, label)

        elif self.experiment=='random_pixels':
           
            return self.static_noise[idx],label
        
        return image,label
        





