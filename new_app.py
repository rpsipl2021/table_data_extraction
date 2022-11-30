# from unittest import result
from predict import PredictFunc
from PIL import Image
from app import TableData
import os
# import cv2

for i in os.listdir('png_images/'):
    image_path = 'png_images/'+ i
    # image_path = './dis1.jpg'
    # print(image_path.split('/')[1].split('.')[0])
    obj = PredictFunc(image_path)
    result = obj.main_inf()
    try:
        x,y,w,h,s = result[0][0]
        img = Image.open(image_path)
        img2 = img.crop((x,y,w,h))
        img2.save("cropped"+"/"+image_path.split('/')[1].split('.')[0]+"_cropped_out.png")
        obj1 = TableData()
        x = obj1.table_data("cropped"+"/"+image_path.split('/')[1].split('.')[0]+"_cropped_out.png")
        data1 = x.to_excel('xlsxfile'+'/'+image_path.split('/')[1].split('.')[0]+'.xlsx',header = False, index = False)
    except:
        print('NO TABLE DETECTED')
