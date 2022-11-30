# # import os
# # from deskew import deskew
# # import glob

# # imagepath = 'total_invoice1/'
# # # rotated_image = []
# # for files in os.listdir(imagepath):
# #     # print(files)
# #     deskew(os.path.join(imagepath) + files)

import cv2
import imutils
import numpy as np
image_path = 'png_images/Scan-3-22-page0.jpg'
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
dist1 = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
dist2 = (dist1 * 255).astype("uint8")
dist3 = cv2.threshold(dist2, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cv2.imshow(' ',dist3)
cv2.waitKey(0)