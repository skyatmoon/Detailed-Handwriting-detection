from __future__ import division, print_function, absolute_import
import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
from torchvision import transforms, datasets
from capsulelayers import DenseCapsule, PrimaryCapsule
import matplotlib.pyplot as plt
from utils import combine_images,save_image # in utils.py save_image is the function that can change one-channel tensor to channels, but still black and white
from PIL import Image
import cv2
import os.path
import glob
import numpy as np
from capsulenet import CapsuleNet

def draw_mask_edge_on_image_cv2(image, mask, color=(0, 0, 255), num = ""):
    coef = 255 if np.max(image) < 3 else 1
    image = (image * coef).astype(np.float32)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image, contours, -1, color, 1)
    #cv2.imshow('test',image)
    cv2.imwrite(args + num + "_" + "/real_and_recon.png", image)
    #cv2.waitKey(0)

def show_reconstruction(model, img, n_images, args, num):

    model.eval()
    x = img
    
    _, x_recon = model(x)
    data_o = np.concatenate([x.cpu().data])
    data_r = np.concatenate([x_recon.cpu().data])
    data = np.concatenate([data_r])

    img = combine_images(np.transpose(data_o, [0, 2, 3, 1]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args + str(num)+ "_" +"real.png")


    img = combine_images(np.transpose(data_r, [0, 2, 3, 1]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args + str(num)+ "_" + "recon.png")

    rel = cv2.imread(args + str(num)+ "_" + "real.png", 0)
    mask = cv2.imread(args + str(num)+ "_" + "recon.png", 0)
    #draw_mask_edge_on_image_cv2(rel, mask, str(num))
        
    img = combine_images(np.transpose(data, [0, 2, 3, 1]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args + str(num)+ "_" + "recon.png")
        # print()
        # print('Reconstructed images are saved to %s/real_and_recon_color.png' % args.save_dir)
        # print('-' * 70)
        # plt.figure()
        # plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png", )) #matplot imshow() WILL AUTO CHANGE ONE CHANNEL IMG TO COLOR ONE
    plt.show()
    


if __name__ == "__main__":
    import argparse
    import os

    model = CapsuleNet(input_size=[1, 28, 28], classes=62, routings=3)
    #model = CapsuleNet(input_size=[1, 28, 28], classes=10, routings=3)
    model.cuda()
    print(model)
    model.load_state_dict(torch.load('/home/skyatmoon/COMP4550/My_project/Caps-model/trained_model-1.pkl'))

    def trans_to_cuda(variable):
        if torch.cuda.is_available():
            return variable.cuda()
        else:
            return variable
        
    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")

    model = trans_to_cuda(model)


    args = './character-level/'
    savepath = './reconstruction-level/'
    input_path = args
    image_paths = []

    if os.path.isdir(input_path):
        for inp_file in os.listdir(input_path):
            image_paths += [input_path + inp_file]
    else:
        image_paths += [input_path]

    image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]
    num = 0

    print("Begin Recon...")
    for img_path in image_paths:
        img = Image.open(img_path).convert('L')
        if img.size[0] != 28 or img.size[1] != 28:
            img = img.resize((28, 28))
        arr = []
        img2 = Image.new('L', (28, 28), 255)
        for i in range(28):
            for j in range(28):

                pixel = 1 - float(img.getpixel((j, i)))/255.0

                arr.append(float(pixel))

        arr1 = np.array(arr).reshape((1, 1,28, 28))
        data = torch.from_numpy(arr1)
        data = data.type(torch.FloatTensor)
        data = data.to(device)
        show_reconstruction(model, data, 1,savepath, num)
        num = num +1
    print("End...")