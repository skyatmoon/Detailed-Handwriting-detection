import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import dataloader
from torchvision import transforms, datasets
from capsulelayers import DenseCapsule, PrimaryCapsule
from datadownloade import EMNIST
import matplotlib.pyplot as plt
from utils import combine_images,save_image # in utils.py save_image is the function that can change one-channel tensor to channels, but still black and white
from PIL import Image
import numpy as np
import cv2
import os.path
import glob
from mydataload import DealDataset

def load_mymnist(path='./data/mymnist', batch_size=2):

    kwargs = {'num_workers': 1, 'pin_memory': True}

    testDataset = DealDataset(path, "train-images-idx3-ubyte.gz","train-labels-idx1-ubyte.gz",transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()]) )
    trainDataset = DealDataset(path, "test-images-idx3-ubyte.gz","train-labels-idx1-ubyte.gz",transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()]) )

    # 训练数据和测试数据的装载
    train_loader = torch.utils.data.DataLoader(
        dataset=trainDataset,
        batch_size=batch_size, 
        shuffle=False,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=testDataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


if __name__ == "__main__":
    import argparse
    import os

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.0005 * 784, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")  # num_routing should > 0
    parser.add_argument('--shift_pixels', default=2, type=int,
                        help="Number of pixels to shift at most in each direction.")
    parser.add_argument('--data_dir', default='./data/mymnist',
                        help="Directory of data. If no data, use \'--download\' flag to download it")
    parser.add_argument('--download', action='store_true',
                        help="Download the required data.")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    #train_loader, test_loader = load_mnist(args.data_dir, download=True, batch_size=args.batch_size)
    #train_loader, test_loader = load_emnist(args.data_dir, download=True, batch_size=args.batch_size)
    train_loader, test_loader = load_mymnist(args.data_dir,batch_size=args.batch_size)

    for data, target in enumerate(train_loader):
	    print(data.size(),target.size())
	    break