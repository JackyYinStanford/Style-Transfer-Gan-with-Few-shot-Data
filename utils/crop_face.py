# -*- coding: utf-8 -*-
# !/usr/bin/env python

import cv2
import numpy as np

FACE_MEAN2 = [-0.094971, 0.151144, -0.093360, 0.227065, -0.089294, 0.304166, -0.081319, 0.381586, -0.069193, 0.459182,
              -0.052547, 0.535981, -0.032875, 0.611261, -0.008049, 0.685606, 0.024174, 0.756265, 0.064731, 0.822912,
              0.111910, 0.884825, 0.164065, 0.942702, 0.220029, 0.997631, 0.278108, 1.050231, 0.341786, 1.094799,
              0.415014, 1.123616, 0.500000, 1.132070, 0.584986, 1.123616, 0.658214, 1.094799, 0.721892, 1.050231,
              0.779971, 0.997631, 0.835935, 0.942702, 0.888090, 0.884825, 0.935269, 0.822912, 0.975826, 0.756265,
              1.008049, 0.685606, 1.032875, 0.611261, 1.052547, 0.535981, 1.069193, 0.459182, 1.081319, 0.381586,
              1.089294, 0.304166, 1.093360, 0.227065, 1.094971, 0.151144, -0.007016, 0.062466, 0.062117, -0.006624,
              0.159912, -0.025746, 0.260402, -0.011624, 0.357498, 0.018106, 0.642502, 0.018106, 0.739598, -0.011624,
              0.840088, -0.025746, 0.937883, -0.006624, 1.007015, 0.062466, 0.500000, 0.214352, 0.500000, 0.326287,
              0.500000, 0.440114, 0.500000, 0.552338, 0.372690, 0.627825, 0.430461, 0.639674, 0.500000, 0.652266,
              0.569539, 0.639674, 0.627310, 0.627825, 0.098222, 0.210955, 0.154045, 0.176009, 0.283686, 0.191279,
              0.326661, 0.241235, 0.267546, 0.256580, 0.147374, 0.246798, 0.673339, 0.241235, 0.716314, 0.191279,
              0.845955, 0.176009, 0.901778, 0.210955, 0.852627, 0.246798, 0.732454, 0.256580, 0.081003, 0.047476,
              0.173043, 0.043127, 0.264444, 0.054412, 0.354234, 0.072408, 0.645766, 0.072408, 0.735556, 0.054412,
              0.826957, 0.043127, 0.918997, 0.047476, 0.220403, 0.171435, 0.206627, 0.260818, 0.217234, 0.214726,
              0.779597, 0.171435, 0.793373, 0.260818, 0.782765, 0.214726, 0.413330, 0.227794, 0.586670, 0.227794,
              0.366299, 0.485450, 0.633701, 0.485450, 0.324881, 0.570364, 0.675119, 0.570364, 0.277690, 0.781648,
              0.359495, 0.759397, 0.443554, 0.748754, 0.500000, 0.754422, 0.556447, 0.748754, 0.640505, 0.759397,
              0.722310, 0.781648, 0.670297, 0.862589, 0.594328, 0.920659, 0.500000, 0.936438, 0.405672, 0.920659,
              0.329703, 0.862589, 0.300374, 0.787910, 0.400305, 0.792216, 0.500000, 0.800824, 0.599695, 0.792216,
              0.699626, 0.787910, 0.604309, 0.833463, 0.500000, 0.849886, 0.395691, 0.833463, 0.217234, 0.214726,
              0.782765, 0.214726]


def parse_pts(file):
    with open(file) as f_p:
        content = list(f_p)
        pts = np.ones((83, 2), dtype=float)
        for i in range(0, 83):
            p_t = content[i].strip().split(' ')
            pts[i, 0] = float(p_t[0])
            pts[i, 1] = float(p_t[2].strip())
    return pts


face_mean = parse_pts(
    '/data/model.pts')


def change_mean(face_mean, alpha=0.1, bias=0.5):
    for index, p_v in enumerate(face_mean):
        p_v[0] = (p_v[0] + bias) * alpha - 1.40
        p_v[1] = (p_v[1] + bias) * alpha - 1.86
        face_mean[index] = p_v

    return face_mean


face_mean = change_mean(face_mean, alpha=0.005, bias=100)


HOMO_MATRIX = np.array([[0, 0, 1]], dtype=np.float32)

LEYE_INDEX = [52, 53, 72, 54, 55, 56, 73, 57]
REYE_INDEX = [61, 60, 75, 59, 58, 63, 76, 62]
LEYE_MEAN = [
    0.0909091,
    0.466438,
    0.313061,
    0.327367,
    0.577138,
    0.309165,
    0.828978,
    0.388135,
    1.,
    0.586939,
    0.764747,
    0.648006,
    0.522315,
    0.664871,
    0.286513,
    0.609078]
LEYE_MEAN = np.array(LEYE_MEAN, dtype=np.float32).reshape(-1, 2)
REYE_MEAN = [
    0.909091,
    0.466438,
    0.686939,
    0.327367,
    0.422862,
    0.309165,
    0.171022,
    0.388135,
    0.,
    0.586939,
    0.235253,
    0.648006,
    0.477685,
    0.664871,
    0.713487,
    0.609078]
REYE_MEAN = np.array(REYE_MEAN, dtype=np.float32).reshape(-1, 2)


def get_best_affine(srcpts, dstpts):
    # srcpts = srcpts.ravel();dstpts = dstpts.ravel();srcpts = srcpts / 540.0
    assert srcpts.size == dstpts.size
    src = srcpts.reshape(-1, 2)
    dst = dstpts.reshape(-1, 2)

    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src_norm = src - src_mean
    dst_norm = dst - dst_mean
    norm = np.linalg.norm(src_norm.ravel(), 2) ** 2
    # print(norm)
    a_1 = np.dot(src_norm.ravel(), dst_norm.ravel()) / norm
    b_1 = (np.dot(src_norm[:, 0], dst_norm[:, 1]) -
         np.dot(src_norm[:, 1], dst_norm[:, 0])) / norm
    return np.array([
        [a_1, -b_1, dst_mean[0] - a_1 * src_mean[0] + b_1 * src_mean[1]],
        [b_1, a_1, dst_mean[1] - b_1 * src_mean[0] - a_1 * src_mean[1]]
    ], dtype=np.float32)


def get_crop_face_best_affine(srcpts, s_z, margin):
    # assert srcpts.size == 212
    mat = get_best_affine(srcpts, face_mean)
    return add_margin_to_affine_mat(mat, s_z, margin)


def crop_face_v2(
        src,
        marks,
        margin=0.15,
        inter_mode=cv2.INTER_CUBIC,
        border_mode=cv2.BORDER_CONSTANT):
    assert src is not None, 'image load error'
    assert marks.shape == (106, 2) or marks.shape == (280, 2)
    if marks.shape[0] == 280:
        marks106 = marks[:106, :]
    else:
        marks106 = marks
    s_x, s_y, sy_out = 192, 288, 256
    marginx = margin
    marginy = get_equal_margin_y(s_x, s_y, margin)
    aff = get_crop_face_best_affine(marks106, (s_x, s_y), (marginx, marginy))
    dst = cv2.warpAffine(src, aff, (s_x, sy_out),
                         flags=inter_mode, borderMode=border_mode)
    return dst, aff


# bbox:x, y, bboxW, bboxW
# out_wh:outW, outH
def crop_face_best(src, srcpts, s_z, margin, bbox=(95, 35, 256, 360), out_wh=(
        320, 320), inter_mode=cv2.INTER_CUBIC, border_mode=cv2.BORDER_CONSTANT):
    mat = get_crop_face_best_affine(srcpts, s_z, margin)
    crop_mat = get_crop_affine_matrix(bbox=bbox, out_wh=out_wh)
    mat_new = combine_affine_and_crop_matrix(mat, crop_mat)
    if isinstance(s_z, int):
        s_z = (s_z, s_z)
    dst = cv2.warpAffine(
        src,
        mat,
        s_z,
        flags=inter_mode,
        borderMode=border_mode)
    dst_crop = cv2.warpAffine(
        src,
        mat_new,
        out_wh,
        flags=inter_mode,
        borderMode=border_mode)
    return dst, dst_crop, mat_new


def crop_face_margin_hair_best(
        src,
        srcpts,
        s_z,
        margin,
        inter_mode=cv2.INTER_CUBIC,
        border_mode=cv2.BORDER_CONSTANT):
    mat = get_crop_face_best_affine(srcpts, s_z, margin)
    # mat_new = add_margin_to_affine_mat(mat, s_z, )
    if isinstance(s_z, int):
        s_z = (s_z, s_z)
    dst = cv2.warpAffine(
        src,
        mat,
        s_z,
        flags=inter_mode,
        borderMode=border_mode)
    # dst_crop = cv2.warpAffine(src, mat_new, s_z, flags = inter_mode, borderMode = border_mode)
    return dst, mat


def crop_hair_crop_best(
        src,
        srcpts,
        s_z,
        box,
        margin,
        inter_mode=cv2.INTER_CUBIC,
        border_mode=cv2.BORDER_CONSTANT):
    mat = get_crop_face_best_affine(srcpts, s_z, margin)
    crop_mat = get_crop_affine_matrix(bbox=box, out_wh=(box[2], box[3]))
    # crop_mat = get_crop_affine_matrix(bbox=box,out_wh=(256,256))
    mat_new = combine_affine_and_crop_matrix(mat, crop_mat)
    if isinstance(s_z, int):
        s_z = (s_z, s_z)
    new_s_z = (box[2], box[3])
    dst = cv2.warpAffine(
        src,
        mat_new,
        new_s_z,
        flags=inter_mode,
        borderMode=border_mode)
    # dst = cv2.warpAffine(src, mat_new,(256,256), flags = inter_mode, borderMode = border_mode)
    return dst, mat_new


def crop_face_best_v1(
        src,
        srcpts,
        s_z,
        margin,
        inter_mode=cv2.INTER_CUBIC,
        border_mode=cv2.BORDER_CONSTANT):
    mat = get_crop_face_best_affine(srcpts, s_z, margin)
    if isinstance(s_z, int):
        s_z = (s_z, s_z)
    dst = cv2.warpAffine(
        src,
        mat,
        s_z,
        flags=inter_mode,
        borderMode=border_mode)
    # dst = cv2.warpAffine(dst, crop_mat, (256,256), flags = inter_mode, borderMode = border_mode)
    return dst, mat


def get_equal_margin_y(s_x, s_y, marginx):
    scale = s_x / (1.0 + 2.0 * marginx)
    return (s_y / scale - 1.0) / 2.0


def add_margin_to_affine_mat(mat, s_z, margin):
    assert mat.shape == (2, 3)
    assert isinstance(s_z, int) or (isinstance(s_z, tuple) and len(s_z) == 2)
    assert isinstance(
        margin,
        float) or (
        isinstance(
            margin,
            tuple) and len(margin) == 2)
    if isinstance(s_z, int):
        s_z = (s_z, s_z)
    if isinstance(margin, float):
        margin = (margin, margin)
    scale = s_z[0] / (1.0 + 2.0 * margin[0]), s_z[1] / (1.0 + 2.0 * margin[1])
    mat_st = np.array([[scale[0], 0, scale[0] * margin[0]],
                       [0, scale[1], scale[1] * margin[1]]], dtype=np.float32)
    return combine_affine_matrix(mat_st, mat)


def combine_affine_matrix(aff1, aff2):
    assert aff1.shape == (2, 3) and aff2.shape == (2, 3)
    mat1 = np.concatenate((aff1.astype('float32'), HOMO_MATRIX), axis=0)
    mat2 = np.concatenate((aff2.astype('float32'), HOMO_MATRIX), axis=0)
    return np.dot(mat1, mat2)[:2]


def get_crop_affine_matrix(bbox, out_wh):
    input_x, input_y, bbw, bbh = bbox
    outw, outh = out_wh
    mat = np.array([
        [outw / bbw, 0, outw / 2 - (outw / bbw) * (input_x + bbw / 2)],
        [0, outh / bbh, outh / 2 - (outh / bbh) * (input_y + bbh / 2)],
    ], dtype=np.float32)
    return mat


def combine_affine_and_crop_matrix(mat, crop_mat):
    crop_mat = np.concatenate(
        (crop_mat.astype('float32'), HOMO_MATRIX), axis=0)
    mat_new = np.concatenate((mat.astype('float32'), HOMO_MATRIX), axis=0)
    mat_new = cv2.gemm(crop_mat, mat_new, 1, None, 0)

    mat_new = mat_new[:2]
    return mat_new
