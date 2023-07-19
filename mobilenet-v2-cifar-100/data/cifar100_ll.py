#from torchvision.datasets import CIFAR100
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os

class CIFAR100_LL(Dataset):
    def __init__(self, args, root, train, transform, task_id):
        self.args = args
        self.root = root
        if train == 'train':
            data_csv = self.root + f'cifar100_train{task_id}.csv'
        elif train == 'replay':
            data_csv = self.root + f'cifar100_replay{task_id}.csv'
        else:
            data_csv = self.root + f'cifar100_test{task_id}.csv'
        
        
        self.data = pd.read_csv(data_csv)
        
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data.iloc[i]['image'], self.data.iloc[i]['label']
        path = os.path.join(self.args.data_dir, path)
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label
        
        
        
class Data:
    def __init__(self, args, task_id):
        # pin_memory = False
        # if args.gpu is not None:
        pin_memory = False

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        self.loader_train = []
        self.loader_replay = []
        self.loader_test = []

        for i in range(task_id+1):
            
            trainset = CIFAR100_LL(args=args, root=args.csv_dir, train='train', 
                                  transform=transform_train, task_id=i)
            
            train_loader = DataLoader(
                trainset, batch_size=args.train_batch_size, shuffle=True, 
                num_workers=1, pin_memory=pin_memory
                )
        
            self.loader_train.append(train_loader)
            
            replayset = CIFAR100_LL(args=args, root=args.csv_dir, train='replay', 
                                  transform=transform_train, task_id=i)
            
            replay_loader = DataLoader(
                replayset, batch_size=args.train_batch_size, shuffle=True, 
                num_workers=1, pin_memory=pin_memory
                )
        
            self.loader_replay.append(replay_loader)
            
            testset = CIFAR100_LL(args=args, root=args.csv_dir, train='test', 
                                 transform=transform_test, task_id=i)
            
            test_loader = DataLoader(
                testset, batch_size=args.eval_batch_size, shuffle=False, 
                num_workers=1, pin_memory=pin_memory)
            
            self.loader_test.append(test_loader)
