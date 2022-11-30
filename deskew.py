import os
import cv2
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import rotate

for i in os.listdir('png_images/'):
  
  image_path = 'png_images/'+ i
  print(image_path)
  img = cv2.imread(image_path)
  sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
  sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
  sobelx1 = cv2.resize(sobelx, (640,640))
  sobely1 = cv2.resize(sobely, (640,640))
  # cv2.imshow('sobelx',sobelx1)
  # cv2.waitKey(0)
  # cv2.imshow('sobely',sobely1)
  # cv2.waitKey(0)

  def horizontal_projections(sobelx):
      return np.sum(sobelx, axis=1)

  # rows,cols = sobelx.shape
  predicted_angle = 0
  highest_hp = 0
  for index,angle in enumerate(range(-10,10)):
    hp = horizontal_projections(rotate(img, angle, cval=1))
    median_hp = np.median(hp)
    print(median_hp)
    if highest_hp < median_hp:
      predicted_angle = angle
      highest_hp = median_hp

  fig, ax = plt.subplots(ncols=2, figsize=(20,10))
  # ax[0].set_title('original image grayscale')
  # ax[0].imshow(img, cmap="gray")
  # ax[0].grid(color='r', linestyle='-', markevery=1)
  # ax[1].set_title('original image rotated by angle'+str(predicted_angle))
  # ax[1].imshow(rotate(img, predicted_angle, cval=1), cmap="gray")
  # #ax[1].grid(color='r', linestyle='-', markevery=1)
  # ax[1].grid(None)
  roatated_img = rotate(img, predicted_angle, cval=1)
  import matplotlib.image
  matplotlib.image.imsave("rotated_fix"+"/"+image_path.split('/')[1].split('.')[0]+".png", roatated_img)