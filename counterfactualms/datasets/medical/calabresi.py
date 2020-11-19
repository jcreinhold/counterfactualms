from torch.utils.data.dataset import Dataset
from skimage.io import imread
import numpy as np
import pandas as pd

import torchvision as tv


class CalabresiDataset(Dataset):
    def __init__(self, csv_path, crop_type=None, crop_size=(192, 192), downsample:int=None):
        super().__init__()
        self.csv_path = csv_path
        self.csv = pd.read_csv(csv_path)
        self.crop_type = crop_type
        self.crop_size = crop_size
        self.downsample = downsample
        self.resize = None if downsample is None else [cs // downsample for cs in self.crop_size]

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        row = self.csv.loc[index]
        img_path = row['filename']
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
        item = self._convert_row(row)
        item['image'] = img
        return item

    @staticmethod
    def _convert_row(row):
        type = {'HC': 0, 'RRMS': 1, 'SPMS': 1}
        sex = {'M': 0, 'F': 1}
        relapse = {np.nan: np.nan, 'N': 0, 'Y': 1}
        out = dict(
            age=row['age'],
            brain_volume=row['brain_volume'],
            duration=row['duration'],
            edss=row['edss'],
            image=None,
            relapse=relapse[row['relapse_last30days']],
            scan=row['scan'],
            sex=sex[row['sex']],
            slice_number=row['slice_number'],
            subject=row['subject'],
            type=type[row['type']],
            ventricle_volume=row['ventricle_volume']
        )
        return out