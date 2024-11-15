import json
import cv2
import numpy as np
from crop_face import crop_face_best


# json文件的点位存储到字典
def get_pts(json_file, size):
    pts_map = {}
    with open(json_file) as file:
        content = file.read()
        data = json.loads(content)
        for d_v in data:
            name = d_v['path']
            try:
                pts = d_v['faces'][0]['face_landmark']
            except KeyError:
                # print(name)
                continue
            pts = np.array(pts).reshape(-1, 2)
            pts = pts * size
            pts_map[name] = pts
    return pts_map


def crop_align(ori, tar, name, pts, ori_size, margin, bbox, output_size):
    pts = pts[0:83]

    if ori is None or tar is None:
        print('input is None')
        return None
        # print(img_cv2.shape)
    if pts is None:
        print('lmk is None')
        return None

    size = ori_size
    _, crop_align_res, affine_mat = crop_face_best(
        tar, pts, size, margin, bbox=bbox)
    align_ori = cv2.warpAffine(
        ori,
        affine_mat,
        (output_size,
         output_size),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT)

    return align_ori, crop_align_res


# 按顺序包住鼻子区域
def get_nose_mask(img, pts, dilate=False, erode=False):
    # input:
    # img: image with correct shape, pts: corresponding facial landmark points
    # dilate: flag to decide whether to dilate, erode: flag to decide whether
    # to erode

    new_pts = np.int32([pts[55:64]])
    new_pts = new_pts.reshape((1, -1, 2))

    mask = np.zeros(img.shape, np.uint8)
    cv2.fillPoly(mask, new_pts, [255, 255, 255])

    if dilate:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    elif erode:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

    return mask


# 按顺序包住鼻孔区域
def get_nostril_mask(img, pts, dilate=False, erode=False):
    # input:
    # img: image with correct shape, pts: corresponding facial landmark points
    # dilate: flag to decide whether to dilate, erode: flag to decide whether
    # to erode

    new_pts = np.int32([pts[56:63]])
    new_pts = new_pts.reshape((1, -1, 2))

    mask = np.zeros(img.shape, np.uint8)
    cv2.fillPoly(mask, new_pts, [255, 255, 255])

    if dilate:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    elif erode:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

    return mask


# 按顺序包住嘴巴区域
def get_mouth_mask(img, pts, dilate=False, erode=False):
    # input:
    # img: image with correct shape, pts: corresponding facial landmark points
    # dilate: flag to decide whether to dilate, erode: flag to decide whether
    # to erode

    new_pts = np.int32([pts[65], pts[67], pts[68], pts[69],
                        pts[70], pts[71], pts[66], pts[79],
                        pts[78], pts[77], pts[76], pts[75]])
    new_pts = new_pts.reshape((1, -1, 2))

    mask = np.zeros(img.shape, np.uint8)
    cv2.fillPoly(mask, new_pts, [255, 255, 255])

    if dilate:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
    elif erode:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

    return mask


# 按顺序包住眼睛区域
def get_eye_mask(img, pts, dilate=False, erode=False):
    # input:
    # img: image with correct shape, pts: corresponding facial landmark points
    # dilate: flag to decide whether to dilate, erode: flag to decide whether
    # to erode

    mask = np.zeros(img.shape, np.uint8)

    left_eye = np.int32([pts[35:43]])
    left_eye = left_eye.reshape((1, -1, 2))
    cv2.fillPoly(mask, left_eye, [255, 255, 255])

    right_eye = np.int32([pts[45:53]])
    right_eye = right_eye.reshape((1, -1, 2))
    cv2.fillPoly(mask, right_eye, [255, 255, 255])

    if dilate:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
    elif erode:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

    return mask


# 按顺序包住眉毛区域
def get_eyebrow_mask(img, pts, dilate=False, erode=False):
    # input:
    # img: image with correct shape, pts: corresponding facial landmark points
    # dilate: flag to decide whether to dilate, erode: flag to decide whether
    # to erode

    mask = np.zeros(img.shape, np.uint8)
    left_eyebrow = np.int32([pts[19:27]])
    left_eyebrow = left_eyebrow.reshape((1, -1, 2))
    cv2.fillPoly(mask, left_eyebrow, [255, 255, 255])

    right_eyebrow = np.int32([pts[27:35]])
    right_eyebrow = right_eyebrow.reshape((1, -1, 2))
    cv2.fillPoly(mask, right_eyebrow, [255, 255, 255])

    if dilate:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
    elif erode:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

    return mask


# 根据点位抠出脸部的轮廓线
def get_face_mask(img, pts, dilate=False, erode=False):
    # input:
    # img: image with correct shape, pts: corresponding facial landmark points
    # dilate: flag to decide whether to dilate, erode: flag to decide whether
    # to erode

    # 增加额头的七个点位
    p_87 = pts[64] + 3.5 * (pts[64] - pts[77])
    p_87 = np.expand_dims(p_87, axis=0)

    p_95 = pts[25] + 1.0 * (pts[25] - pts[37])
    p_95 = np.expand_dims(p_95, axis=0)
    p_96 = pts[19] + 0.6 * (pts[25] - pts[37])
    p_96 = np.expand_dims(p_96, axis=0)
    p_90 = pts[0] + 0.6 * (pts[24] - pts[40])
    p_90 = np.expand_dims(p_90, axis=0)

    p_92 = pts[33] + 1.0 * (pts[33] - pts[47])
    p_92 = np.expand_dims(p_92, axis=0)
    p_93 = pts[27] + 0.6 * (pts[33] - pts[47])
    p_93 = np.expand_dims(p_93, axis=0)
    p_91 = pts[18] + (pts[31] - pts[49])
    p_91 = np.expand_dims(p_91, axis=0)

    p_mouth = pts[77]
    p_nose = (pts[55] + pts[63]) / 2
    direction = p_nose - p_mouth
    forehead_center = (p_nose + 1.6 * direction)  # 1.4
    forehead_center = np.expand_dims(forehead_center, axis=0)
    pts = np.concatenate((pts, forehead_center), axis=0)
    p_brow_left = pts[23]
    p_brow_right = pts[31]
    direction = p_brow_left - p_nose
    forehead_center_left = (p_brow_left + 2.5 * direction)  # 1.9
    direction = p_brow_right - p_nose
    forehead_center_right = (p_brow_right + 2.5 * direction)  # 1.9
    forehead_center_left = np.expand_dims(forehead_center_left, axis=0)
    forehead_center_right = np.expand_dims(forehead_center_right, axis=0)

    pts = np.concatenate((pts[:19], p_91, p_93, p_92,
                          p_87, p_95, p_96, p_90), axis=0)

    mask = np.zeros(img.shape, np.uint8)

    cv2.fillPoly(mask, np.int32([pts]), [255, 255, 255])

    if dilate:
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=5)
    elif erode:
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=3)

    return mask


# 膨胀mask
def dilate_mask(mask, kernel_size, iter):
    # input: mask - original mask
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=iter)
    return mask


# 腐蚀mask
def erode_mask(mask, kernel_size, iter):
    # input: mask - original mask
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=iter)
    return mask
