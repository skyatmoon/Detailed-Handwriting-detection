from __future__ import division, print_function, absolute_import
import numpy as np
from PIL import Image

img = Image.open('./mydata/0.jpg').convert('L')
if img.size[0] != 28 or img.size[1] != 28:
    img = img.resize((28, 28))



arr = []
img2 = Image.new('L', (28, 28), 255)
for i in range(28):
    for j in range(28):

        pixel = 1.0 - float(img.getpixel((j, i)))/255.0

        arr.append(pixel)

arr1 = np.array(arr).reshape((1, 1,28, 28))
print(arr1.shape)
