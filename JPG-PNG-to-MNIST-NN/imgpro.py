from PIL import Image
import cv2 as cv
import os
 
#input_dir = ""#'./training-images/0/'
#out_dir ="" #'./training-images/0/'
#a = os.listdir(input_dir)

for num in range(0,63):
        input_dir = './test-images/'+ str(num) + '/'
        out_dir = './test-images/' + str(num) + '/'
        a = os.listdir(input_dir)   


        for i in a:
                print(i)
                image = cv.imread(input_dir+i)
                gray = cv.cvtColor(image,cv.COLOR_RGB2GRAY)   #要二值化图像，要先进行灰度化处理
                gray = cv.resize(gray,(28,28))
                ret, binary = cv.threshold(gray,127,255,cv.THRESH_BINARY | cv.THRESH_OTSU)
                binary = 255 - binary
                cv.imwrite(out_dir+i,binary)

        '''

        I = Image.open(input_dir+i)
        L = I.convert('L')
        threshold = 200
        
        table = []
        for i in range(256):
                if i < threshold:
                table.append(0)

                
                        
                
        photo = I.point(table, '1')
        photo.save("test2.jpg")
        L.save(out_dir+i)
        '''


