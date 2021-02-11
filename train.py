'''Train script.
'''

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchsummary import summary
from torchvision import transforms

from level_dict import hierarchy
from runtime_args import args
from load_dataset import LoadDataset
from  model import resnet50
from model.hierarchical_loss import HierarchicalLossNetwork
from helper import calculate_accuracy

device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')

train_dataset = LoadDataset(csv_path=args.train_csv, cifar_metafile=args.metafile, transform=transforms.ToTensor())
test_dataset = LoadDataset(csv_path=args.test_csv, cifar_metafile=args.metafile, transform=transforms.ToTensor())

train_generator = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.no_shuffle, num_workers=args.num_workers)
test_generator = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.no_shuffle, num_workers=args.num_workers)

model = resnet50.ResNet50()
optimizer = Adam(model.parameters(), lr=args.learning_rate)


model = model.to(device)

HLN = HierarchicalLossNetwork(metafile_path=args.metafile, hierarchical_labels=hierarchy, device=device)


for epoch_idx in range(args.epoch):

    i = 0
    epoch_loss = 0
    epoch_superclass_accuracy = 0
    epoch_subclass_accuracy = 0
    for i, sample in tqdm(enumerate(train_generator)):


        batch_x, batch_y1, batch_y2 = sample['image'].to(device), sample['label_1'].to(device), sample['label_2'].to(device)
        optimizer.zero_grad()

        superclass_pred,subclass_pred = model(batch_x)
        prediction = [superclass_pred,subclass_pred]
        dloss = HLN.calculate_dloss(prediction, [batch_y1, batch_y2])
        lloss = HLN.calculate_lloss(prediction, [batch_y1, batch_y2])

        total_loss = lloss + dloss
        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item()
        epoch_superclass_accuracy += calculate_accuracy(predictions=prediction[0], labels=batch_y1)
        epoch_subclass_accuracy += calculate_accuracy(predictions=prediction[1], labels=batch_y2)



    print(f'Loss at {epoch_idx} : {epoch_loss/(i+1)}')
    print(f'Superclass accuracy at {epoch_idx} : {epoch_superclass_accuracy/(i+1)}')
    print(f'Subclass accuracy at {epoch_idx} : {epoch_subclass_accuracy/(i+1)}')






