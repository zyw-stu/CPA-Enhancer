import numpy as np
import os
import cv2
import math
import random


def load_annotations(annot_path):
    with open(annot_path, 'r') as f:
        txt = f.readlines()
        annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
    return annotations


def AddHaz_loop(img_f, center, size, beta, A):
    (row, col, chs) = img_f.shape
    for j in range(row):
        for l in range(col):
            d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
            td = math.exp(-beta * d)
            img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
    return img_f


def parse_annotation(annotation, fog_probability):
    line = annotation.split()
    image_path = line[0]
    img_name = image_path.split('/')[-1]
    image_name = img_name.split('.')[0]
    image_name = image_name.split('\\')[-1]
    image_name_index = img_name.split('.')[1]

    if not os.path.exists(image_path):
        raise KeyError("%s does not exist..." % image_path)

    image = cv2.imread(image_path)
    i = random.random()  # Use random.random() to generate a random float in the range [0.0, 1.0)

    # Check if the image should have fog
    if i < fog_probability:
        img_f = image / 255
        (row, col, chs) = image.shape
        A = 0.5
        beta = 0.01 * random.randint(0, 9) + 0.05
        size = math.sqrt(max(row, col))
        center = (row // 2, col // 2)
        foggy_image = AddHaz_loop(img_f, center, size, beta, A)
        img_f = np.clip(foggy_image * 255, 0, 255)
        img_f = img_f.astype(np.uint8)
        img_name = 'G:\\dataset\\detection\\data_hybrid\\hybrid_voc_fog\\train\\JPEGImages\\' + image_name + '.' + image_name_index
        cv2.imwrite(img_name, img_f)
    else :
        img_name = 'G:\\dataset\\detection\\data_hybrid\\hybrid_voc_fog\\train\\JPEGImages\\' + image_name + '.' + image_name_index
        cv2.imwrite(img_name,image)


if __name__ == '__main__':
    an = load_annotations('G:\\project\\dehaze\\Image-Adaptive-YOLO-main\\data\\dataset_fog\\voc_norm_train.txt')
    ll = len(an)
    print(ll)

    # Set fog_probability to 2/3 for processing 2/3 of the images
    fog_probability = 2 / 3

    for j in range(ll):
        print('j', j)
        parse_annotation(an[j], fog_probability)
