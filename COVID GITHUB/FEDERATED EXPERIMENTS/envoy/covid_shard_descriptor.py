# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Covid Shard Descriptor."""

import logging
import os
from typing import List
from torch.utils.data import Dataset, random_split
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import PIL
from PIL import Image
from torchvision import transforms as T



import numpy as np
import requests

from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor

logger = logging.getLogger(__name__)


class CovidShardDataset(ShardDataset):
    """Covid Shard dataset class."""

    def __init__(self, img, tab, y, data_type, rank=1, worldsize=1):
        """Initialize CovidDataset."""
        self.data_type = data_type
        self.rank = rank
        self.worldsize = worldsize
        self.img = img[self.rank - 1::self.worldsize]
        self.tab = tab[self.rank - 1::self.worldsize]
        self.y = y[self.rank - 1::self.worldsize]

    def __getitem__(self, index: int):
        """Return an item by the index."""
        return self.img[index], self.tab[index], self.y[index]

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.img)


class CovidShardDescriptor(ShardDescriptor):
    """Covid Shard descriptor class."""

    def __init__(
            self,
            rank_worldsize: str = '1, 1',
            hospital: str = '',
            **kwargs
    ):
        """Initialize CovidShardDescriptor."""
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))
        self.hospital = hospital
        (img_train, tab_train, y_train), (img_test, tab_test, y_test) = self.download_data()
        self.data_by_type = {
            'train': (img_train, tab_train, y_train),
            'val': (img_test, tab_test, y_test)
        }

    def get_shard_dataset_types(self) -> List[str]:
        """Get available shard dataset types."""
        return list(self.data_by_type)

    def get_dataset(self, dataset_type='train'):
        """Return a shard dataset by type."""
        if dataset_type not in self.data_by_type:
            raise Exception(f'Wrong dataset type: {dataset_type}')
        return CovidShardDataset(
            *self.data_by_type[dataset_type],
            data_type=dataset_type,
            rank=self.rank,
            worldsize=self.worldsize
        )
    
    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return ['1', '256', '256']

    @property
    def target_shape(self):
        """Return the target shape info."""
        return ['1', '256', '256']

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return (f'Images-tabular dataset, shard number {self.rank}'
                f' out of {self.worldsize}')
    
    
    def download_data(self):
        """Download prepared dataset."""
        img_train = []
        tab_train = []
        y_train = []
        img_test = []
        tab_test = []
        y_test = []
        
     
        #print("Pre image data", os.getcwd())
        my_transform = T.Compose([T.Resize((256,256)),
                                  T.RandomApply(
                                    [T.RandomHorizontalFlip(),
                                     T.RandomCrop(256, padding=4)],
                                    p=.5
                                  )])
        #image_data = ImageDataset(excel_file="../train1AllColumns.xls", image_dir="../Trainset", transform=my_transform)
        #test_image_data = ImageDataset(excel_file="../test1AllColumns.xls", image_dir="../Testset", transform=my_transform)
        image_data = ImageDataset(ospedale = self.hospital, excel_file="../trainANDtest.xls", image_dir="../DATASET", transform=my_transform)
        train_size = int(0.80 * len(image_data))
        val_size = int((len(image_data) - train_size))
        image_data, test_image_data = random_split(image_data, (train_size, val_size))
        
        #print("Post image data", os.getcwd())
        print("Hospital number:", self.hospital)
        print("Total training samples:", len(image_data))
        print("Total test samples:", len(test_image_data))

        
        for item in image_data:
            img_train.append(item[0])
            tab_train.append(item[1])
            y_train.append(item[2])
          
        for item in test_image_data:
            img_test.append(item[0])
            tab_test.append(item[1])
            y_test.append(item[2])
        

        return (img_train, tab_train, y_train), (img_test, tab_test, y_test)
    

    
class ImageDataset(Dataset):
    """Tabular and Image dataset."""

    def __init__(self, ospedale, excel_file, image_dir, transform=None):
        self.image_dir = image_dir
        self.ospedale = ospedale
        self.excel_file = excel_file
        self.tabular = pd.read_excel(excel_file)
        self.tabular = self.tabular[self.tabular["Hospital"]==self.ospedale]
        self.transform = transform

    def __len__(self):
        return len(self.tabular)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #se voglio un ospedale in particolare:
        #tabular = tabular[tabular["Hospital"]=="A"]
        tabular = self.tabular.iloc[idx, 0:]
        #print("image dir:", self.image_dir)
        y = tabular["Prognosis"]
        
        #print(os.getcwd())
        os.listdir()
        
        #os.chdir("..")
        #os.chdir("TrainSet")
        image = PIL.Image.open(f"{self.image_dir}/{tabular['ImageFile']}")
        #image = PIL.Image.open(f"{tabular['ImageFile']}")
        image = image.convert('L')
        image = np.array(image)
        #image = image[..., :3]
        
        image = T.functional.to_tensor(image)

        tabular = tabular[['Age', 'Sex', 'PositivityAtAdmission',
       'Temp_C', 'DaysFever', 'Cough', 'DifficultyInBreathing', 'WBC', 'RBC',
       'CRP', 'Glucose', 'LDH', 'INR', 'PaO2', 'PaCO2', 'pH',
       'HighBloodPressure', 'Diabetes', 'Dementia', 'BPCO', 'Cancer',
       'ChronicKidneyDisease', 'RespiratoryFailure']]
        tabular = tabular.tolist()
        tabular = torch.FloatTensor(tabular)
        
        if self.transform:
            image = self.transform(image)

        return image, tabular, y
