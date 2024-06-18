from torchvision import transforms
from torch.utils.data import Dataset
import os
import time
import torch
from collections import Counter
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# from torchvision import transforms, datasets
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

from network import CNN, CNN2, CNN3, CNN4, CNN5, ResNet
from losses import loss
from optimizer import create_optimizer
from engine import train, test
from plot import plot_loss_and_acc

from sklearn.model_selection import train_test_split
# from vit import VisionTransformer
# filter the warnings
import warnings
warnings.filterwarnings("ignore")


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        info = img_name.split('_')
        win_size = int(info[1])
        # stock movement in the next 5 days
        # label 1 refers to up, 0 down
        label5 = (int(info[3]) + 1) / 2
        label10 = (int(info[4]) + 1) / 2
        label15 = (int(info[5]) + 1) / 2
        last_d = info[6]

        if self.transform:
            image = self.transform(image)

        return image, int(label5), img_name


def get_args_parser():
    parser = argparse.ArgumentParser('CNN Training', add_help=False)
    parser.add_argument('--dataset', default='Finance', type=str,
                        help='choose dataset (default: Finance)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size of the dataset default(128)')
    parser.add_argument('--epoch', type=int, default=1,
                        help='epochs of training process default(10)')
    # Optimization parameters
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adam"')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (default: 4e-3), with total batch size 4096')
    parser.add_argument('--loss', type=str, default='ce',
                        help='define loss function (default: CrossEntropy)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='choose running device (default: Cuda)')
    parser.add_argument('--exp', type=str, default='debug',
                        help='choose using mode, (default: experiment mode)')
    parser.add_argument('--model', type=str, default='resnet',
                        help='choosing the model (default: cnn)')
    parser.add_argument('--infer', type=int, default=0,
                        help='if infer mode or not default(0)')
    parser.add_argument('--small_set', type=int, default=0,
                        help='using the small dataset or not default(0)')
    return parser


transform_train = transforms.Compose([
    # transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
    #                              ratio=(3.0/4.0, 4.0/3.0)),
    # transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    # transforms.Resize(img_resize),
    # transforms.CenterCrop(img_resize),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Only perform deterministic operations on image augmentation on the test set
transform_test = transforms.Compose([
    # transforms.Resize(img_resize),
    # transforms.CenterCrop(img_resize),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def main(args):
    # Load the dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Set up the training device
    device = torch.device(args.device)
    # in_chans = 3
    model = CNN4()
    # Define which model to use
    if args.model == 'cnn2':
        model = CNN2()
    if args.model == 'cnn3':
        model = CNN3()
    if args.model == 'cnn4':
        model = CNN4()
    elif args.model == 'cnn5':
        model = CNN5()

    model = model.to(device)
    # Print out model information
    fp = open('output.log', 'a+')
    print(f"using {device} device", file=fp)
    print(f"dataset:{args.dataset}", file=fp)
    print(f"model:{args.model}", file=fp)
    print(f"using {device} device")
    print(f"dataset:{args.dataset}")
    print(f"model:{args.model}")
    fp.close()

    dataset = CustomImageDataset(root_dir=os.path.join(current_dir, 'train'), transform=transform_train)
    test_ds = CustomImageDataset(root_dir=os.path.join(current_dir, 'test'), transform=transform_test)
    train_set, val_set = train_test_split(dataset, test_size=0.1, random_state=42)
    # Define the data loader for training data
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    if args.infer == 1:
        model.load_state_dict(torch.load('my_model.pth'))

    # Define loss function
    criterion = loss(args)
    # Define optimizer
    optimizer = create_optimizer(args, model)

    if not args.infer:
        model_trained, best_model, train_los, train_acc, val_los, val_acc = train(model=model,
                                                                                  criterion=criterion,
                                                                                  train_loader=train_loader,
                                                                                  val_loader=test_loader,
                                                                                  optimizer=optimizer,
                                                                                  device=device,
                                                                                  max_epoch=args.epoch,
                                                                                  disp_freq=100)
    # test
    prob, path = test(model=best_model,
                      criterion=criterion,
                      test_loader=test_loader,
                      device=device)
    prob = np.concatenate(prob, axis=0)
    print(prob.shape)
    print(path[0])
    date = [p.split('_')[6][:len('xxxx-xx-xx')] for p in path]
    df = pd.DataFrame(date, columns=['Date'])
    df['Prob'] = prob
    df.to_csv(os.path.join(current_dir, 'prob_output.csv'))
    # Plotting loss and accuracy
    fp = open('output.log', 'a+')
    print(f'Drawing...', file=fp)
    print(f'Drawing...')

    # Plotting loss and accuracy
    suffix1 = args.dataset + '_' + args.model + '_lr' + str(args.lr) + '_' + str(
        args.opt) + '_wd' + str(args.weight_decay) + '_epoch' + str(args.epoch) + '.png'
    # path1 = ['train_loss_' + suffix1, 'train_acc_' + suffix1]
    path1 = ['loss_' + suffix1, 'acc_' + suffix1]
    suffix2 = args.dataset + '_' + args.model + '_lr' + str(args.lr) + '_' + str(
        args.opt) + '_wd' + str(args.weight_decay) + '_epoch' + str(args.epoch) + '.png'
    path2 = ['val_loss_' + suffix2, 'val_acc_' + suffix2]
    prefix = args.dataset + '_' + args.model + '_lr' + str(args.lr) + '_' + str(
        args.opt) + '_wd' + str(args.weight_decay) + '_epoch' + str(args.epoch)
    if not args.infer:
        # Save the model
        torch.save(best_model.state_dict(), prefix + '_model.pth')
        '''
        plot_loss_and_acc({'TRAIN': [train_los, train_acc]}, path1)
        plot_loss_and_acc({'VAL': [val_los, val_acc]}, path2)
        '''
        plot_loss_and_acc({'model_train': [train_los, train_acc]}, {'model_val':  [val_los, val_acc]}, path1)
        print("Draw Done", file=fp)
        print("Draw Done")
    fp.close()


if __name__ == "__main__":
    """
    Main code
    """
    parser = argparse.ArgumentParser('Prombelm solver', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
