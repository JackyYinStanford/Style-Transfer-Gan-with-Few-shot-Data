# flip侧脸
import os
import cv2
import tqdm


def generate_weights(
        training_set_dir,
        pose_file,
        flip_hair_dir,
        de_norm=False):

    training_set = os.listdir(training_set_dir)
    pose_file = open(pose_file, 'r')
    pose_dic = {}
    for line in pose_file.readlines():
        values = line.rstrip('\n').split(' ')
        name = values[0].split('/')[-1]
        pose_dic[name] = [
            float(
                values[1]), float(
                values[2])]  # name : [yaw, pitch]

    cnt = 0
    for name in tqdm.tqdm(training_set[:]):
        if 'beard_' in name:
            search_name = name.replace('beard_', '')
        elif 'moustache_s1_' in name:
            search_name = name.replace('moustache_s1_', '')
        elif 'moustache_s2_' in name:
            search_name = name.replace('moustache_s2_', '')
        elif 'moustache_s3_' in name:
            search_name = name.replace('moustache_s3_', '')
        elif 'glass_' in name:
            search_name = name.replace('glass_', '')
        elif 'flip_' in name:
            search_name = name.replace('flip_', '')
        else:
            search_name = name

        # 是否要反归一化
        if search_name in pose_dic:
            yaw = pose_dic[search_name][0]
        else:
            print('无pose数据: ', str(search_name))
            continue
        if de_norm:
            yaw = yaw * 100

        # 需要flip的图片
        if yaw >= 15 or yaw <= -15:
            if os.path.exists(flip_hair_dir + '/flip_' + search_name):
                # if 'flip_' + search_name in os.listdir(flip_hair_dir):
                if not os.path.isfile(
                        training_set_dir + '/flip_' + search_name):
                    print('flip_' + search_name)
                    cnt += 1
                    #                 print(cnt)
                    img = cv2.imread(
                        os.path.join(
                            training_set_dir,
                            search_name))
                    img_a = cv2.flip(img[:, 0:256, :], 1, dst=None)
                    img_b = cv2.flip(img[:, 256:, :], 1, dst=None)
                    img_flip = cv2.hconcat([img_a, img_b])
                    #                     if cnt < 20:
                    #                         plt.imshow(img_flip[:,:,::-1])
                    #                         plt.show()
                    cv2.imwrite(
                        training_set_dir +
                        '/flip_' +
                        search_name,
                        img_flip)
    print(cnt)


IMG_DIR = '/data/pairs_for_distill/Cycle-StyleGAN-VGG_sample_eye/train/'
POSE_TXT = '/data/pose_add_big_pose.txt'
FLIP_HAIR_DIR = '/data/pairs_for_distill' \
                '/hair_recon/hair_for_train_filler'

generate_weights(IMG_DIR, POSE_TXT, FLIP_HAIR_DIR, section=9, de_norm=False)
