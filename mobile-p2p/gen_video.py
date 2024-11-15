import os
import glob
import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from skimage import io


def gen_video(exp_name, path):
    name = exp_name
    filename = path.split('/')[-1]

    os.makedirs(path.replace(filename, 'output_video'), exist_ok=True)

    files = glob.glob(path + '/*.jpg')
    print('nums of pictures : ' + str(len(files)))

    fps = 25
    size = (540 * 2, 960)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    video = cv2.VideoWriter(path.replace(filename, 'output_video/') + name + '.avi', fourcc, fps, size)
    files.sort()
    for f in (files):
        img = cv2.imread(f)
        img = cv2.resize(img, size)
        video.write(img)
    video.release()

if __name__ == '__main__':
    exp_name = 'teen_small_320_13k_ngf16_v3'
    # 执行inference输出fc结果
    # 执行compositeFrames输出每一帧
    # result_dir = compositeFrames(exp_name=exp_name)
    # 执行gen_video合成视频
    result_dir = '/Users/jackyyin/Documents/data/变年轻/video_test/teen_small_320_13k_ngf16_v3'
    gen_video(exp_name, result_dir)