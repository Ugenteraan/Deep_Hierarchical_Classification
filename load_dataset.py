'''Pytorch dataset loading script.
'''

import os
import pickle
import csv
import cv2
from PIL import Image
from torch.utils.data import Dataset
from level_dict import hierarchy
from helper import read_meta


class LoadDataset(Dataset):
    '''Reads the given csv file and loads the data.
    '''


    def __init__(self, csv_path, cifar_metafile, image_size=32, image_depth=3, return_label=True, transform=None):
        '''Init param.
        '''

        assert os.path.exists(csv_path), 'The given csv path must be valid!'

        self.csv_path = csv_path
        self.image_size = image_size
        self.image_depth = image_depth
        self.return_label = return_label
        self.meta_filename = cifar_metafile
        self.transform = transform
        self.data_list = self.csv_to_list()
        self.coarse_labels, self.fine_labels = read_meta(self.meta_filename)

        #check if the hierarchy dictionary is consistent with the csv file
        for k,v in hierarchy.items():
            if not k in self.coarse_labels:
                print(f"Superclass missing! {k}")
            for subclass in v:
                if not subclass in self.fine_labels:
                    print(f"Subclass missing! {subclass}")



    def csv_to_list(self):
        '''Reads the path of the file and its corresponding label
        '''

        with open(self.csv_path, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)

        return data


    def __len__(self):
        '''Returns the total amount of data.
        '''
        return len(self.data_list)

    def __getitem__(self, idx):
        '''Returns a single item.
        '''
        image_path, image, superclass, subclass = None, None, None, None
        if self.return_label:
            image_path, superclass, subclass = self.data_list[idx]
        else:
            image_path = self.data_list[idx]

        if self.image_depth == 1:
            image = cv2.imread(image_path, 0)
        else:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.image_size != 32:
            cv2.resize(image, (self.image_size, self.image_size))


        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        if self.return_label:
            return {
                'image':image/255.0,
                'label_1': self.coarse_labels.index(superclass.strip(' ')),
                'label_2': self.fine_labels.index(subclass.strip(' '))
            }
        else:
            return {
                'image':image
            }

