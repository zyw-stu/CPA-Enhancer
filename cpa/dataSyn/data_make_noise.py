import numpy as np
import os
import cv2
import math
# from numba import jit
import random

# only use the image including the labeled instance objects for training
def load_annotations(annot_path): # 从voc_norm_train.txt文件中加载标注信息
    # print(annot_path)
    with open(annot_path, 'r') as f:
        txt = f.readlines()
        annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
    return annotations # 每一行代表一张图片，格式为 图片路径， xmin, ymin, xmax, ymax ,class_idx, ( xmin, ymin, xmax, ymax ,class_idx,...)

def Noise_loop(img_f, sigma):
    noise = np.random.randn(*img_f.shape)
    noisy_patch = img_f + noise * sigma
    return noisy_patch
# print('*****************Add Noise offline***************************')
def parse_annotation(annotation): #

    line = annotation.split()
    image_path = line[0] # 文件的整体路径
    # print(image_path)
    img_name = image_path.split('/')[-1] # 文件名，包含后缀
    # print('img_name',img_name)
    image_name = img_name.split('.')[0] # 文件名，不含后缀
    image_name=image_name.split('\\')[-1]
    # print('image_name',image_name)
    image_name_index = img_name.split('.')[1] # 文件名的后缀
    # print('image_name_index',image_name_index)

    #'/data/vdd/liuwenyu/data_vocfog/train/JPEGImages/'
    if not os.path.exists(image_path):
        raise KeyError("%s does not exist ... " %image_path)
    image = cv2.imread(image_path) # 打开文件
    img_d = image / 255  # 归一化

    sigma=random.uniform(15/255,50/255)
    noise_image=Noise_loop(img_d,sigma)

    img_d=np.clip(noise_image*255, 0, 255) # 限制范围在(0,255)内
    img_d = img_d.astype(np.uint8)
    img_name = 'G:\\dataset\\detection\\data_vocnoise\\test\\JPEGImages\\' + image_name + '.' + image_name_index
    cv2.imwrite(img_name, img_d)



if __name__ == '__main__':
    # an = load_annotations('/home/liuwenyu.lwy/code/defog_yolov3/data/dataset/voc_norm_train.txt')
    an = load_annotations('G:\\project\\dehaze\\Image-Adaptive-YOLO-main\\data\\dataset_fog\\voc_norm_test.txt')
    #an = load_annotations('/home/liuwenyu.lwy/code/defog_yolov3/data/dataset/voc_norm_test.txt')
    ll = len(an)
    print(ll)
    for j in range(ll):
        print('j',j)
        parse_annotation(an[j])

