# 局部先美白再锐化
import os
from tqdm import tqdm
from PIL import Image, ImageFilter
from get_facial_masks import get_pts, get_mouth_mask
import numpy as np
import cv2


class MYLUT:
    def __init__(self, lutpath='./face.png'):
        lut = cv2.imread(lutpath)
        #         print(lut.shape)
        cube64rows = 8
        cube64size = 64
        # cube256rows = 16
        cube256size = 256
        cube_scale = cube256size // cube64size  # 4
        reshape_lut = np.zeros((cube256size, cube256size, cube256size, 3))
        for i in range(cube64size):
            c_x = (i % cube64rows) * cube64size
            c_y = (i // cube64rows) * cube64size
            #             print(cy,cy + cube64size, cx,cx + cube64size)
            cube64 = lut[c_y:c_y + cube64size, c_x:c_x + cube64size]
            _rows, _cols, _ = cube64.shape
            if _rows == 0 or _cols == 0:
                continue
            cube256 = cv2.resize(cube64, (cube256size, cube256size))
            i = i * cube_scale
            for k in range(cube_scale):
                reshape_lut[i + k] = cube256
        self.lut = reshape_lut

    def image_lut(self, src):
        arr = src.copy()
        b_s = arr[:, :, 0]
        g_s = arr[:, :, 1]
        r_s = arr[:, :, 2]
        arr[:, :] = self.lut[b_s, g_s, r_s]  # numpy写的越骚，运行速度越快
        return arr


ROOT = '/data/fake_smile_5w/train_sr3/'
GEN_ROOT = '/data/fake_smile_5w/train_B_sr_sharpen3/'
SAVE_ROOT = '/data/fake_smile_5w/train_sr_whiten_sharpen/'
# train_sr_whiten_sharpen是先美白，再用反锐化掩码滤波去锐化

os.makedirs(SAVE_ROOT, exist_ok=True)

pts_map = get_pts(
    json_file='/data/fake_smile_5w/4w_sr_sharpen3.json')

COUNT = 0
files = os.listdir(ROOT)

lut = MYLUT()

for (k, v) in tqdm(pts_map.items()):

    if k not in files:
        continue
    COUNT += 1
    if COUNT == 20:
        break
    name = k
    gen_path = GEN_ROOT + name
    save_path = SAVE_ROOT + name

    if os.path.exists(save_path):
        continue

    img = cv2.imread(ROOT + name)

    h, w, c = img.shape
    w2 = int(w / 2)
    ori_face = img[0:h, 0:w2]
    ori_img = img[0:h, w2:w]

    gen_img = lut.image_lut(ori_img)

    mask, mask_img = get_mouth_mask(v, gen_img)

    if mask is None:
        continue

    mask = cv2.GaussianBlur(mask, (7, 7), 10)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask = mask * 0.3

    whiten_img = mask / 255.0 * gen_img + (1 - mask / 255.0) * ori_img

    whiten_img = whiten_img.astype('uint8')

    # OpenCV转换成PIL.Image格式，因为要使用Image的滤波器
    whiten_img = Image.fromarray(cv2.cvtColor(whiten_img, cv2.COLOR_BGR2RGB))
    sharpen_whiten_img = whiten_img.filter(
        ImageFilter.UnsharpMask(
            radius=2, percent=500, threshold=3))

    sharpen_whiten_img = cv2.cvtColor(
        np.array(sharpen_whiten_img),
        cv2.COLOR_RGB2BGR)  # PIL.Image转换成OpenCV格式

    res = mask / 255.0 * sharpen_whiten_img + (1 - mask / 255.0) * ori_img

    res = res.astype('uint8')

    res = np.hstack((ori_face, res))

    cv2.imwrite(save_path, res, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
