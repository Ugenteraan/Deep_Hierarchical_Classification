'''Pre-processing script to read Cifar-100 dataset and write the images onto disk with the corresponding labels recorded in a csv file.
'''

import os
import pickle
import numpy as np
import pandas as pd
import imageio
import cv2
from tqdm import tqdm
from helper import unpickle, read_meta


class Preprocess_Cifar100:
    '''Process the pickle files.
    '''


    def __init__(self, meta_filename='./dataset/pickle_files/meta', train_file='./dataset/pickle_files/train', test_file='./dataset/pickle_files/test',
                        image_write_dir='./dataset/images/', csv_write_dir='./dataset/', train_csv_filename='train.csv', test_csv_filename='test.csv'):
        '''Init params.
        '''
        self.meta_filename = meta_filename
        self.train_file = train_file
        self.test_file = test_file
        self.image_write_dir = image_write_dir
        self.csv_write_dir = csv_write_dir
        self.train_csv_filename = train_csv_filename
        self.test_csv_filename = test_csv_filename

        if not os.path.exists(self.image_write_dir):
            os.makedirs(self.image_write_dir)

        if not os.path.exists(self.csv_write_dir):
            os.makedirs(self.csv_write_dir)

        self.coarse_label_names, self.fine_label_names = read_meta(meta_filename=self.meta_filename)



    def process_data(self, train=True):
        '''Read the train/test data and write the image array and its corresponding label into the disk and a csv file respectively.
        '''

        if train:
            pickle_file = unpickle(self.train_file)
        else:
            pickle_file = unpickle(self.test_file)

        filenames = [t.decode('utf8') for t in pickle_file[b'filenames']]
        fine_labels = pickle_file[b'fine_labels']
        coarse_labels = pickle_file[b'coarse_labels']
        data = pickle_file[b'data']


        filenames = [t.decode('utf8') for t in pickle_file[b'filenames']]
        fine_labels = pickle_file[b'fine_labels']
        coarse_labels = pickle_file[b'coarse_labels']
        data = pickle_file[b'data']

        images = []
        for d in data:
            image = np.zeros((32,32,3), dtype=np.uint8)
            image[:,:,0] = np.reshape(d[:1024], (32,32))
            image[:,:,1] = np.reshape(d[1024:2048], (32,32))
            image[:,:,2] = np.reshape(d[2048:], (32,32))
            images.append(image)

        if train:
            csv_filename = self.train_csv_filename
        else:
            csv_filename = self.test_csv_filename

        with open(f'{self.csv_write_dir}/{csv_filename}', 'w+') as f:
            for i, image in enumerate(images):
                filename = filenames[i]
                coarse_label = self.coarse_label_names[coarse_labels[i]]
                fine_label = self.fine_label_names[fine_labels[i]]
                imageio.imsave(f'{self.image_write_dir}{filename}', image)
                f.write(f'{self.image_write_dir}{filename}, {coarse_label}, {fine_label}\n')



p = Preprocess_Cifar100()
p.process_data(train=True) #process the training set
p.process_data(train=False) #process the testing set
