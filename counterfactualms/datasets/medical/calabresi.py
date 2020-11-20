import numpy as np
import pandas as pd
from skimage.io import imread
import torch
from torch.utils.data.dataset import Dataset
import torchvision as tv


class CalabresiDataset(Dataset):
    def __init__(self, csv_path, crop_type=None, crop_size=(192, 192), downsample:int=None):
        super().__init__()
        self.csv_path = csv_path
        csv = pd.read_csv(csv_path)
        csv.drop(['fss', 'msss', 'treatment_propagated', 'treatment'], axis=1, inplace=True)
        csv.rename(columns={'relapse_last30days': 'relapse'}, inplace=True)
        csv['relapse'] = csv['relapse'].map({np.nan: -1., 'N': 0., 'Y': 1.})
        csv['duration'] = csv['duration'].map({np.nan: 0.})
        csv['edss'] = csv['edss'].map({np.nan: -1.})
        csv['sex'] = csv['sex'].map({'M': 0., 'F': 1.})
        csv['type'] = csv['type'].map({'HC': 0., 'RRMS': 1., 'SPMS': 1., 'PPMS': 1.})
        self.csv = csv
        self.crop_type = crop_type
        self.crop_size = crop_size
        self.downsample = downsample
        self.resize = None if downsample is None else [cs // downsample for cs in self.crop_size]

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        item = self.csv.loc[index]
        item = item.to_dict()
        item = self._prepare_item(item)
        img_path = item['filename']
        img = imread(img_path, as_gray=True)

        transform_list = []
        transform_list += [tv.transforms.ToPILImage()]
        if self.crop_type is not None:
            if self.crop_type == 'center':
                transform_list += [tv.transforms.CenterCrop(self.crop_size)]
            elif self.crop_type == 'random':
                transform_list += [tv.transforms.RandomCrop(self.crop_size)]
            else:
                raise ValueError(f'unknown crop type: {self.crop_type}')

        if self.resize:
            transform_list += [tv.transforms.Resize(self.resize)]

        transform_list += [tv.transforms.ToTensor()]
        img = tv.transforms.Compose(transform_list)(img)
        item['image'] = img
        return item

    @staticmethod
    def _prepare_item(item):
        item['age'] = torch.as_tensor([item['age']], dtype=torch.float32)
        item['sex'] = torch.as_tensor([item['type']], dtype=torch.float32)
        item['type'] = torch.as_tensor([item['type']], dtype=torch.float32)
        item['relapse_last30days'] = torch.as_tensor([item['relapse_last30days']], dtype=torch.float32)
        item['duration'] = torch.as_tensor([item['duration']], dtype=torch.float32)
        item['brain_volume'] = torch.as_tensor(['brain_volume'], dtype=torch.float32)
        item['ventricle_volume'] = torch.as_tensor(['ventricle_volume'], dtype=torch.float32)
        item['edss'] = torch.as_tensor(['edss'], dtype=torch.float32)
        return item