import numpy as np
import os
import cv2
import math
import random


def load_annotations(annot_path): 
    # print(annot_path)
    with open(annot_path, 'r') as f:
        txt = f.readlines()
        annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
    return annotations

def get_noise(img, value=10):

    noise = np.random.uniform(0, 256, img.shape[0:2])
    v = value * 0.01
    noise[np.where(noise < (256 - v))] = 0

    k = np.array([[0, 0.1, 0],
                  [0.1, 8, 0.1],
                  [0, 0.1, 0]])

    noise = cv2.filter2D(noise, -1, k)
    return noise


def rain_blur(noise, length=10, angle=0, w=1):
    trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
    dig = np.diag(np.ones(length))
    k = cv2.warpAffine(dig, trans, (length, length))
    k = cv2.GaussianBlur(k, (w, w), 0)
    blurred = cv2.filter2D(noise, -1, k)
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred


def alpha_rain(rain, img, beta=0.8):
    rain = np.expand_dims(rain, 2)
    rain_effect = np.concatenate((img, rain), axis=2)  # add alpha channel
    rain_result = img.copy()  
    rain = np.array(rain, dtype=np.float32)  
    rain_result[:, :, 0] = rain_result[:, :, 0] * (255 - rain[:, :, 0]) / 255.0 + beta * rain[:, :, 0]
    rain_result[:, :, 1] = rain_result[:, :, 1] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    rain_result[:, :, 2] = rain_result[:, :, 2] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    return rain_result


def parse_annotation(annotation):

    line = annotation.split()
    image_path = line[0] 
    # print(image_path)
    img_name = image_path.split('/')[-1] 
    # print('img_name',img_name)
    image_name = img_name.split('.')[0] 
    image_name=image_name.split('\\')[-1]
    # print('image_name',image_name)
    image_name_index = img_name.split('.')[1] 
    # print('image_name_index',image_name_index)

    if not os.path.exists(image_path):
        raise KeyError("%s does not exist ... " %image_path)
    image = cv2.imread(image_path) 
    value=random.randint(300,600)
    noise = get_noise(image, value=500)
    rain = rain_blur(noise, length=50, angle=-30, w=3)
    rain_result= alpha_rain(rain, image, beta=0.6)  

    img_d = rain_result / 255  
    img_d=np.clip(img_d*255, 0, 255) 
    img_d = img_d.astype(np.uint8)
    img_name = 'G:\\dataset\\detection\\data_vocrain\\train\\JPEGImages\\' + image_name + '.' + image_name_index
    cv2.imwrite(img_name, img_d)


if __name__ == '__main__':
    an = load_annotations('G:\\project\\dehaze\\Image-Adaptive-YOLO-main\\data\\dataset_fog\\voc_norm_train.txt')
    ll = len(an)
    print(ll)
    for j in range(ll):
        print('j',j)
        parse_annotation(an[j])

