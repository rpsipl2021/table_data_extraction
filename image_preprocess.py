import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import rotate

class preprocses():
    def  __init__(self) -> None:
        pass
    def binary(self, image_path):
        img = cv2.imread(image_path, 0)
        ret, imgf = cv2.threshold(img, 0 ,255, cv2.THRESH_BINARY, cv2.THRESH_OTSU)
            
        
