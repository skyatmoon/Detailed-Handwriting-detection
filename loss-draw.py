import torch.backends.cudnn as cudnn
import math
import time
import datetime
import argparse
import os
from torch.nn import CTCLoss
#from vision.network import CRNN
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
#from util.data_loader import RegDataSet
#from util.tools import *
#from IAMloade import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file_path1 = "/home/skyatmoon/COMP4550/****CapsNet-Pytorch-master/v-l/"  
file_path2 = "/home/skyatmoon/COMP4550/****CapsNet-Pytorch-master/t-l/"  
file_path3 = "/home/skyatmoon/COMP4550/****CapsNet-Pytorch-master/acc/"  
path_list1 = os.listdir(file_path1) 
path_list2 = os.listdir(file_path2) 
path_list3 = os.listdir(file_path3) 

path_list1.sort(key= lambda x:float(x[5:]))
path_list2.sort(key= lambda x:float(x[5:]))
path_list3.sort(key= lambda x:float(x[5:]))

epoch=np.loadtxt('enum')


for i in range(0,100):
    trainloss=np.loadtxt("/home/skyatmoon/COMP4550/****CapsNet-Pytorch-master/t-l/" + path_list2[i])
    valloss=np.loadtxt("/home/skyatmoon/COMP4550/****CapsNet-Pytorch-master/v-l/" + path_list1[i])
    #acc=np.loadtxt("/home/skyatmoon/COMP4550/****CapsNet-Pytorch-master/acc/" + path_list3[i])

    plt.plot(epoch, valloss, color = "r", marker = ".")
    # plt.xlabel("epoch")
    # plt.ylabel("loss")
    # plt.title("val-loss")
    # plt.savefig("val-loss.png")


    plt.plot(epoch, trainloss, color="b", marker = ".")

    #plt.plot(epoch, acc, color="g", marker = ".")
    
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("acc-train-val-loss")
    plt.savefig("val-train-loss" + "-" + str(i) + ".png")
    plt.clf()