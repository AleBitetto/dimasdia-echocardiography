import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

def null_collate(batch):
    batch_size = len(batch)
    images = np.array([x[0] for x in batch])
    images = torch.from_numpy(images)
    
    labels = np.array([x[1] for x in batch])
    labels = torch.from_numpy(labels)
    labels = labels.unsqueeze(1)
    
    indices = np.array([x[2] for x in batch])
    indices = torch.from_numpy(indices)
    indices = indices.unsqueeze(1)

    assert(images.shape[0] == labels.shape[0] == indices.shape[0] == batch_size)
    
    return images, labels, indices

class SubsetRandomDataLoader(DataLoader):
    def __init__(self, dataset, indexes, batch_size):
        loader_params = dict(
            batch_size=batch_size,
            num_workers=1,
            pin_memory=True,
            collate_fn=null_collate,
            sampler = SubsetRandomSampler(indexes)
        )
        super(SubsetRandomDataLoader, self).__init__(dataset=dataset, **loader_params)
