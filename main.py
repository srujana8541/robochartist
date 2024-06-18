import os
import time
import torch
import torchvision
from collections import Counter
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from torchvision import transforms, datasets
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


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def get_args_parser():
    parser = argparse.ArgumentParser('CNN Training', add_help=False)
    parser.add_argument('--dataset', default='Finance', type=str,
                        help='choose dataset (default: Finance)')
    parser.add_argument('--batch_size', type=int, default=2,
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
    parser.add_argument('--device', type=str, default='cuda',
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
    # Randomly crop the image to get an image with an area of ​​0.08 to 1 times the original image area 
    # and a height-to-width ratio of 3/4 to 4/3, 
    # and then scale it down to a new image with a height and width of 224 pixels.
    # transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
    #                              ratio=(3.0/4.0, 4.0/3.0)),
    
    # Randomly flip horizontally with probability 0.5
    # transforms.RandomHorizontalFlip(),
    
    # Randomly change brightness, contrast, and saturation
    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    # transforms.Resize(img_resize),

    # Crop out a square area with a height and width of 224 in the center of the image
    # transforms.CenterCrop(img_resize),

    transforms.ToTensor(),
    
    # Normalize each channel
    # (0.485, 0.456, 0.406) and (0.229, 0.224, 0.225) are the mean and variance of each channel calculated on ImageNet.
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
])


#Only perform deterministic operations on image augmentation on the test set
transform_test = transforms.Compose([
    # transforms.Resize(img_resize),

    # Crop out a square area with a height and width of 224 in the center of the image
    # transforms.CenterCrop(img_resize),

    transforms.ToTensor(),

     # Normalize each channel
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def main(args):
    # Load the dataset
    # TODO: change the path to new folder
    new_data_dir = "/home/wenh/ecometrics/data_MA"
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

    '''
    elif args.model == 'resnet':
        model = ResNet(in_chans=in_chans, img_H=32, img_W=32)
    elif args.vit == 'vit':
        model = VisionTransformer(img_size=32, patch_size=2, num_classes=10, num_heads=1, depth=1, embed_dim=32)   
    '''

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

    dataset = datasets.ImageFolder(root=os.path.join(new_data_dir, 'train'), transform=transform_train)

    # test_ds = datasets.ImageFolder(root=os.path.join(new_data_dir, 'test'), transform=transform_train)
    test_ds = ImageFolderWithPaths(os.path.join(new_data_dir, 'test'), transform=transform_train)
    train_set, val_set = train_test_split(dataset, test_size=0.001, random_state=42)
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

    # TODO: can be deleted?
    # prob, path = test(model=model,
    #                   criterion=criterion,
    #                   test_loader=test_loader,
    #                   device=device)

    # train
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
    path = [p.split('/')[-1][:-len('.png')].split('-') for p in path]
    df = pd.DataFrame(path, columns=['SecuCode', 'Tradingday'])
    df['Prob'] = prob
    df.to_csv('./prob_output.csv')
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
