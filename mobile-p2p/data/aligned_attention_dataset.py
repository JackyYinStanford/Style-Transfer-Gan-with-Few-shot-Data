import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torchvision.transforms as transforms
import numpy as np
import cv2
import tqdm


class AlignedAttentionDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

            
        self.AB_size = len(self.AB_paths)
        
        self.mean = [0.5, 0.5, 0.5] #[0.485, 0.456, 0.406]
        self.std = [0.5, 0.5, 0.5] #[0.229, 0.224, 0.225]
        osize = [448, 448]

        if True:

            size = (opt.crop_size, opt.crop_size)

            tf1 = transforms.Compose([
                transforms.Pad(8, fill=0, padding_mode='constant'),
                transforms.RandomCrop(size, pad_if_needed=True),
            ])
            
            scale_ratio = 1.0
            DEGREE = 2 * scale_ratio
            RATE = 0.02 * scale_ratio
            tf2 = transforms.RandomChoice([
                transforms.RandomAffine(degrees=DEGREE, translate=(RATE, RATE), scale=(1.0 - RATE, 1.0 + RATE),
                                        resample= Image.BICUBIC),
                transforms.RandomAffine(degrees=DEGREE, translate=(RATE, RATE), scale=(1.0 - RATE, 1.0 + RATE), 
                                        resample= Image.BICUBIC),
                transforms.RandomAffine(degrees=DEGREE, translate=(RATE, RATE), scale=(1.0 - RATE, 1.0 + RATE),
                                        resample= Image.BICUBIC),
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
                                           #transforms.RandomHorizontalFlip(),
                                           tf1,
                                           transforms.RandomApply([tf2], 0.5),                                          
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.5], std=[0.5])
                                          ])

        #assert(opt.resize_or_crop == 'none')
        self.AB_len = len(self.AB_paths)
        if self.opt.forDebug:
            # For Debug
            self.AB_len = 200 #1280

        self.mask_template = Image.new('RGB', size, (0, 0, 0))
        if self.opt.load_memory:
            
            self.A_images = [1]*self.AB_len
            self.B_images = [1]*self.AB_len
            
            
            for index in tqdm.tqdm(range(self.AB_len)):
                AB_img = Image.open(self.AB_paths[index]).convert('RGB')

                w, h = AB_img.size
                w2 = int(w / 2)
                self.A_images[index] = AB_img.crop((0, 0, w2, h))
                self.B_images[index] = AB_img.crop((w2, 0, w, h))

                
    def __getitem__(self, index):
        
        if self.opt.forDebug:
            # For Debug
            index = index%200  #1280
            
        if self.opt.load_memory:
            if index == 0:
                seed = np.random.randint(2147483647)
                random.seed(seed)
                random.shuffle(self.A_images)
                random.seed(seed)
                random.shuffle(self.B_images)
                    
            A_img = self.A_images[index]
            B_img = self.B_images[index]
            AB_path = self.AB_paths[index]
                
        else:
            if index == 0:
                seed = np.random.randint(2147483647)
                random.seed(seed)
                random.shuffle(self.AB_paths)
 
            AB_path = self.AB_paths[index % self.AB_size]
            AB_img = Image.open(AB_path).convert('RGB')

            w, h = AB_img.size
            w2 = int(w / 2)
            A_img = AB_img.crop((0, 0, w2, h))
            B_img = AB_img.crop((w2, 0, w, h))
            
            # if hair_path is None:
            #     print('shape........', mask_path)


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

        # if self.opt.use_mask:
        #     return {'A': A, 'B': B, 'Mask': self.mask,
        #             'A_paths': A_path, 'B_paths': B_path}
        # else:
        res = {'A': A, 'B': B, 'AB_paths': AB_path}
        return res

    def __len__(self):
        return (self.AB_size)

    def name(self):
        return 'AlignedAttentionDataset'
