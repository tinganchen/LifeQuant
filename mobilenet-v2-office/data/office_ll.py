from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os

class office_simple_LL(Dataset):
    def __init__(self, args, root, train, transform, task_id):
        self.args = args
        self.root = root
        if train == 'train':
            data_csv = self.root + f'train{task_id}.csv'
        elif train == 'replay':
            data_csv = self.root + f'replay{task_id}.csv'
        else:
            data_csv = self.root + f'test{task_id}.csv'
        
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
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
        self.loader_train = []
        self.loader_replay = []
        self.loader_test = []
        
        for i in range(task_id+1):
            
            trainset = office_simple_LL(args=args, root=args.csv_dir, train='train', 
                                  transform=transform_train, task_id=i)
            
            train_loader = DataLoader(
                trainset, batch_size=args.train_batch_size, shuffle=True, 
                num_workers=2, pin_memory=pin_memory
                )
        
            self.loader_train.append(train_loader)
            
            replayset = office_simple_LL(args=args, root=args.csv_dir, train='replay', 
                                  transform=transform_train, task_id=i)
            
            replay_loader = DataLoader(
                replayset, batch_size=args.train_batch_size, shuffle=True, 
                num_workers=2, pin_memory=pin_memory
                )
        
            self.loader_replay.append(replay_loader)
        
            testset = office_simple_LL(args=args, root=args.csv_dir, train='test', 
                                 transform=transform_test, task_id=i)
            
            test_loader = DataLoader(
                testset, batch_size=args.eval_batch_size, shuffle=False, 
                num_workers=2, pin_memory=pin_memory)
            
            self.loader_test.append(test_loader)
