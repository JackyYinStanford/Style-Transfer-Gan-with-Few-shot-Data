# real a和real b同时加眼睛数据
import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

files = glob.glob(
    '/data/aligned_dataset/glasses_face/*')
glasses_files = glob.glob(
    '/data/glasses_seg_data/ori/*')

SAVE_ROOT = '/data/aligned_dataset/glasses_face_3k'
os.makedirs(SAVE_ROOT, exist_ok=True)

mask_all = np.zeros((1024, 1024, 3), dtype='uint8')
image_all = np.zeros((1024, 1024, 3), dtype='uint8')

TOP = 360
WIDTH = 0
for index, f in tqdm(enumerate(files)):

    name = f.split('/')[-1]
    save_path = os.path.join(SAVE_ROOT, name)

    #     if os.path.exists(save_path):
    #         continue

    try:
        real_a = cv2.imread(f)
        h, w, c = real_a.shape

        glasses_path = glasses_files[index % len(glasses_files)]

        glasses_seg_path = glasses_path.replace(
            '/ori/', '/mask_new/').replace('.jpg', '.png')
        glasses_seg = cv2.imread(glasses_seg_path) * 255
        glasses_seg = glasses_seg.astype('uint8')

        glasses_img = cv2.imread(glasses_path)
        glasses_img = cv2.resize(glasses_img, (int(1.1 * 1024), 512))
        glasses_seg = cv2.resize(glasses_seg, (int(1.1 * 1024), 512))
        glasses_img = cv2.resize(glasses_img, (int(720), int(288)))
        glasses_img = cv2.copyMakeBorder(
            glasses_img, 0, 0, 152, 152, cv2.BORDER_CONSTANT, value=0)
        glasses_seg = cv2.resize(glasses_seg, (int(720), int(288)))
        glasses_seg = cv2.copyMakeBorder(
            glasses_seg, 0, 0, 152, 152, cv2.BORDER_CONSTANT, value=0)

        mask_all[TOP:TOP + 288, :, :] = glasses_seg
        image_all[TOP:TOP + 288, :, :] = glasses_img
        mask_all2 = mask_all[:, 0:0 + 1024, :]
        image_all2 = image_all[:, 0:0 + 1024, :]

        real_a = image_all2 * (mask_all2 / 255.0) + \
            real_a * (1 - mask_all2 / 255.0)

        res = real_a
        res = res.astype('uint8')

        if index == 0:
            print(save_path)
        ret = cv2.imwrite(save_path, res, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        if not ret:
            print(f.split('/')[-1])

        if index % 200 == 0:
            res = res[:, :, ::-1]
            plt.figure("Image")
            plt.imshow(res)
            plt.axis('on')
            plt.show()

    except TypeError:
        print("{} Failed".format(f.split('/')[-1]))
