from PIL import Image
from math import sqrt
import numpy as np
from numpy import save


imag = Image.open("project/wash_turb.png")
#Convert the image te RGB if it is a .gif for example
imag = imag.convert ('RGB')
#coordinates of the pixel
X,Y = 0,0
#Get RGB
pixelRGB = imag.getpixel((X,Y))
R,G,B = pixelRGB 

arr = np.zeros((1001,1001))



for y in range(imag.size[0]):
    for x in range(imag.size[1]):
        pixelRGB = imag.getpixel((x,y))
        R,G,B = pixelRGB 
        value = (R + G + B)/(3*255)
        arr[x][y] = value

save("data_draw.npy", arr)