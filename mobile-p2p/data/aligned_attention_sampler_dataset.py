import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torchvision.transforms as transforms
import numpy as np
import cv2
import tqdm
import pdb


class AlignedAttentionSamplerDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths

        if opt.use_sampler:
            self.weights_List = self.generate_weights(self.opt.ratio)

        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        self.AB_size = len(self.AB_paths)

        self.mean = [0.5, 0.5, 0.5]  # [0.485, 0.456, 0.406]
        self.std = [0.5, 0.5, 0.5]  # [0.229, 0.224, 0.225]

        size = (256, 256)

        tf1 = transforms.Compose([
            transforms.Pad(8, fill=0, padding_mode='constant'),
            transforms.RandomCrop(size, pad_if_needed=True),
        ])

        scale_ratio = 1.0
        DEGREE = 2 * scale_ratio
        RATE = 0.02 * scale_ratio
        tf2 = transforms.RandomChoice([
            transforms.RandomAffine(degrees=DEGREE, translate=(RATE, RATE), scale=(1.0 - RATE, 1.0 + RATE),
                                    resample=Image.BICUBIC),
            transforms.RandomAffine(degrees=DEGREE, translate=(RATE, RATE), scale=(1.0 - RATE, 1.0 + RATE),
                                    resample=Image.BICUBIC),
            transforms.RandomAffine(degrees=DEGREE, translate=(RATE, RATE), scale=(1.0 - RATE, 1.0 + RATE),
                                    resample=Image.BICUBIC),
        ])

        RATE_COLOR = 0.04
        tf3 = transforms.ColorJitter(brightness=RATE_COLOR, contrast=RATE_COLOR, saturation=RATE_COLOR,
                                     hue=RATE_COLOR / 2.0)

        self.input_tf = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            tf1,
            transforms.RandomApply([tf2], 0.5),
            tf3,
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        self.mask_tf = transforms.Compose([
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            tf1,
            transforms.RandomApply([tf2], 0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # assert(opt.resize_or_crop == 'none')
        self.AB_len = len(self.AB_paths)
        if self.opt.forDebug:
            # For Debug
            self.AB_len = 200  # 1280

        self.mask_template = Image.new('RGB', size, (0, 0, 0))
        if self.opt.load_memory:

            self.A_images = [1] * self.AB_len
            self.B_images = [1] * self.AB_len

            for index in tqdm.tqdm(range(self.AB_len)):
                AB_img = Image.open(self.AB_paths[index]).convert('RGB')

                w, h = AB_img.size
                w2 = int(w / 2)
                self.A_images[index] = AB_img.crop((0, 0, w2, h))
                self.B_images[index] = AB_img.crop((w2, 0, w, h))

                if self.opt.use_feather:
                    # 边缘羽化
                    feather_mask = np.zeros((256, 256, 3), np.uint8)
                    super_box = feather_mask.copy()
                    super_box[10:246, 10:246] = 255
                    feather_mask = np.where(super_box == np.array([255, 255, 255]), super_box, feather_mask)
                    feather_mask = cv2.GaussianBlur(feather_mask, (121, 121), 0)
                    feather_mask = feather_mask / 255.0
                    feather_mask = np.power(feather_mask, self.opt.feather_ratio)
                    self.B_images[index] = self.B_images[index] * feather_mask + (1 - feather_mask) * self.A_images[index]
                    self.B_images[index] = self.B_images[index].astype(np.uint8)

    def generate_weights(self, ratio):    # 控制采样率， 值越小越容易采到样本分布少的脸

        def get_label_yaw(yaw_list):

            label_yaw_list = []
            label_len = 0
            for yaw in yaw_list:
                # yaw ranges from -39 to 39.5
                if yaw < 0:
                    yaw = -1.0 * yaw
                label_yaw = yaw // 5.0
                # len is 8, range from [0,7]
                label_yaw_list.append(label_yaw)
                label_len = max(label_len, int(label_yaw + 1))
            return label_yaw_list, label_len

        # training_set = os.listdir(self.dir_AB)
        training_set = [name.split('/')[-1] for name in self.AB_paths]
        male_file = open('/data/crop4/no_crop/male_for_train_pose.txt', 'r')
        female_file = open('/data/crop4/no_crop/female_for_train_pose.txt', 'r')
        profile_file = open('/data/hair_0709/tgt3/output-0709_tgt.txt', 'r')

        pose_file = male_file.readlines() + female_file.readlines() + profile_file.readlines()

        pose_dic = {}
        for line in pose_file:
            values = line.rstrip('\n').split(' ')
            name = values[0].split('/')[-1]
            pose_dic[name] = [float(values[1]), float(values[2])]  # name : [yaw, pitch]

        yaw_list = []

        for name in training_set:
            # pose dic里的图片都是以png结尾的
            name = name[:-3] + 'png'
            if name.startswith('glass_'):
                name = name.replace('glass_', '')
            if name not in pose_dic.keys():
                print(name)
                continue
            yaw_list.append(pose_dic[name][0])

        assert len(yaw_list) == len(training_set)

        label_yaw_list, label_len = get_label_yaw(yaw_list)
        lower = 2000  # 低频
        upper = 3500  # 高频
        # ratio = 0.8
        # pdb.set_trace()
        weights = [0] * label_len  # 共有8个label

        for label in label_yaw_list:
            weights[int(label)] += 1  # 统计各个label的数量
        print("The number of images of each label are {}".format(weights))

        for i in range(len(weights)):
            if weights[i] <= lower:  # 低频直接锁死
                weights[i] = lower
            elif weights[i] <= upper:
                weights[i] = int((weights[i] - lower // 2) * ratio) + lower
            else:
                weights[i] = int((weights[i] - upper) * ratio * ratio * ratio) + lower + (upper - lower) * ratio
        if self.opt.average_sample: 
            weights_p = [round(1 / len(weights), 3) for i in range(len(weights))]
        elif self.opt.sample_weights:
            weights_p = self.opt.sample_weights
        else:
            weights_p = [round(num / sum(weights), 3) for num in weights]
        print('weights of each label is ', weights_p)
        weights_List = [0] * len(self.AB_paths)
        for i in range(len(weights_List)):
            weights_List[i] = weights_p[int(label_yaw_list[i])]

        return weights_List

    def __getitem__(self, index):
        size = (256, 256)
        if self.opt.forDebug:
            # For Debug
            index = index % 200  # 1280

        if self.opt.load_memory:
            # noShuffle
            # if index == 0:
            #     seed = np.random.randint(2147483647)
            #     random.seed(seed)
            #     random.shuffle(self.A_images)
            #     random.seed(seed)
            #     random.shuffle(self.B_images)

            A_img = self.A_images[index]
            B_img = self.B_images[index]
            AB_path = self.AB_paths[index]

        else:
            # noShuffle
            # if index == 0:
            #     seed = np.random.randint(2147483647)
            #     random.seed(seed)
            #     random.shuffle(self.AB_paths)

            AB_path = self.AB_paths[index % self.AB_size]
            AB_img = Image.open(AB_path).convert('RGB')

            w, h = AB_img.size
            w2 = int(w / 2)
            A_img = AB_img.crop((0, 0, w2, h))
            B_img = AB_img.crop((w2, 0, w, h))


        seed = np.random.randint(2147483647)
        random.seed(seed)
        A = self.input_tf(A_img)
        random.seed(seed)
        B = self.input_tf(B_img)
        random.seed(seed)

        if self.opt.direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        res = {'A': A, 'B': B, 'AB_paths': AB_path}
        return res

    def __len__(self):
        return (self.AB_size)

    def name(self):
        return 'AlignedAttentionSamplerDataset'
