'''Train script.
'''

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchsummary import summary
from torchvision import transforms

from runtime_args import args
from load_dataset import LoadDataset
from  model import resnet50

device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')

train_dataset = LoadDataset(csv_path=args.train_csv, cifar_metafile=args.metafile, transform=transforms.ToTensor())
test_dataset = LoadDataset(csv_path=args.test_csv, cifar_metafile=args.metafile, transform=transforms.ToTensor())

train_generator = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.no_shuffle, num_workers=args.num_workers)
test_generator = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.no_shuffle, num_workers=args.num_workers)

model = resnet50.ResNet50()

model = model.to(device)


# for i, sample in tqdm(enumerate(train_generator)):

#     batch_x, batch_y1, batch_y2 = sample['image'].to(device), sample['label_1'].to(device), sample['label_2'].to(device)
#     print(batch_y1, batch_y2)

#     a,b = model(batch_x)
#     # print(a,b)

