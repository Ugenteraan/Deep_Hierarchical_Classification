'''Configurations
'''

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--train_csv', type=str, help='Specify the path to the train csv file.', default='./dataset/train.csv')
parser.add_argument('--test_csv', type=str, help='Specify the path to the test csv file.', default='./dataset/test.csv')
parser.add_argument('--metafile', type=str, help='Specify the path to the test csv file.', default='./dataset/pickle_files/meta')
parser.add_argument('--model_save_path', type=str, help='Specify the path to save the model.', default='./')
parser.add_argument('--graphs_folder', type=str, help='Specify the path to save the graphs.', default='./graph_folder/')
parser.add_argument('--epoch', type=int, help='Specify the number of epochs for the training.', default=100)
parser.add_argument('--batch_size', type=int, help='Specify the batch size to be used during training/testing.', default=10)
parser.add_argument('--num_classes', type=int, help='Specify the number of classes the dataset has.', default=6)
parser.add_argument('--learning_rate', type=float, help='Specify the batch size to be used during training.', default=1e-4)
parser.add_argument('--dropout_rate', type=float, help='Specify the dropout rate to be used during training.', default=0.5)
parser.add_argument('--num_workers', type=int, help='Specify the number of workers to be used to load the data.', default=4)
parser.add_argument('--no_shuffle', help='Use this flag to disable shuffling during data loading', action='store_false')
parser.add_argument('--img_size', type=int, help='Specify the size of the input image.', default=32)
parser.add_argument('--img_depth', type=int, help='Specify the depth of the input image.', default=3)
parser.add_argument('--device', type=str, help='Specify which device to be used for the evaluation. Either "cpu" or "gpu".', default='gpu')

args = parser.parse_args()