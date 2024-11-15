import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torchvision.transforms as transforms
import numpy as np
#from opencv_transforms import opencv_transforms as transforms
#from opencv_transforms import transforms 
import cv2


class ffhqSRDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        BaseDataset.__init__(self, opt)
        # A: Blur B: Clear
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        # self.hair_paths = [path.replace('/'+opt.phase,'/hair') for path in self.AB_paths]
        #self.hair_mask_paths = [path.replace('/'+opt.phase,'/hairseg') for path in self.AB_paths]
            
        self.AB_size = len(self.AB_paths)
        # self.hair_size = len(self.hair_paths)
        #self.hair_mask_size = len(self.hair_mask_paths)
        
#         assert self.AB_size <= self.hair_size
        #assert self.AB_size <= self.hair_mask_size

        # self.transform = get_transform(opt)
        #self.use_mask = opt.use_mask
        self.mean = [0.5, 0.5, 0.5] #[0.485, 0.456, 0.406]
        self.std = [0.5, 0.5, 0.5] #[0.229, 0.224, 0.225]
        osize = [256, 256]
        # Data Augmentation
        if True:

            size = (256, 256)

            tf1 = transforms.Compose([
                transforms.Pad(8, fill=0, padding_mode='constant'),
                transforms.RandomCrop(size, pad_if_needed=True),
            ])
            
            # change from 1.0
            scale_ratio = 3.0
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

            # change from 0.04
            RATE_COLOR = 0.12
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

        #assert(opt.resize_or_crop == 'none')
        self.AB_len = len(self.AB_paths)
        # For Debug
        # self.AB_len = 1280
        #self.mask_template = Image.new('RGB', size, (255, 255, 255))
        # self.mask_template = Image.new('RGB', size, (0, 0, 0))
        if self.opt.load_memory:
            
            self.A_images = [1]*self.AB_len
            self.B_images = [1]*self.AB_len
            # self.hair_images = [1]*self.AB_len
            # self.hair_mask_images = [1]*self.AB_len
            import tqdm
            
            
            for index in tqdm.tqdm(range(self.AB_len)):
                AB_img = Image.open(self.AB_paths[index]).convert('RGB')
                self.A_images[index] = AB_img.resize((128, 128), Image.BILINEAR)
                self.A_images[index] = self.A_images[index].resize((256, 256), Image.BILINEAR)

                self.B_images[index] = AB_img.resize((256, 256), Image.BILINEAR)
                # Hair_img = Image.open(self.AB_paths[index]).convert('RGB')
                # w, h = AB_img.size
                # w2 = int(w / 2)
                # self.A_images[index] = AB_img.crop((0, 0, w2, h))
                # self.B_images[index] = AB_img.crop((w2, 0, w, h))

                # hair_image = Image.open(self.hair_paths[index]).convert('RGB')
                # w, h = hair_image.size
                # w2 = int(w / 2)
                # self.hair_mask_images[index] = hair_image.crop((0, 0, w2, h))
                # self.hair_images[index] = hair_image.crop((w2, 0, w, h))                
                # self.hair_mask_images[index] = self.hair_mask_images[index].convert('L')
                
                # self.hair_images[index] = self.hair_images[index].resize(size)
                # self.hair_mask_images[index]= self.hair_mask_images[index].resize(size)
                # self.hair_images[index] = Image.composite(self.hair_images[index],self.mask_template,self.hair_mask_images[index])

                
    def __getitem__(self, index):
        size = (256,256)
        # For Debug
        # index = index%1280
        if self.opt.load_memory:
            if index == 0:
                seed = np.random.randint(2147483647)
                random.seed(seed)
                random.shuffle(self.A_images)
                random.seed(seed)
                random.shuffle(self.B_images)
                # random.seed(seed)
                # random.shuffle(self.hair_images)
                # random.seed(seed)
                # random.shuffle(self.hair_mask_images)
                    
            A_img = self.A_images[index]
            B_img = self.B_images[index]
            # hair_image = self.hair_images[index]
            # hair_mask_image = self.hair_mask_images[index]
            AB_path = self.AB_paths[index]
                
        else:
            if index == 0:
                seed = np.random.randint(2147483647)
                random.seed(seed)
                random.shuffle(self.AB_paths)
                random.seed(seed)
                random.shuffle(self.hair_paths)
 
            AB_path = self.AB_paths[index % self.AB_size]
            AB_img = Image.open(AB_path).convert('RGB')
            hair_path = self.hair_paths[index % self.AB_size]

            w, h = AB_img.size
            w2 = int(w / 2)
            A_img = AB_img.crop((0, 0, w2, h))
            B_img = AB_img.crop((w2, 0, w, h))
            
            hair_image = Image.open(hair_path).convert('RGB')
            w, h = self.hair_image
            w2 = int(w / 2)
            hair_mask_image = hair_image.crop((0, 0, w2, h))
            hair_image = hair_image.crop((w2, 0, w, h))     
            
            hair_mask_image = hair_mask_image.convert('L')            
            hair_mask_image = hair_mask_image.resize(size)
            
            hair_image = hair_image.resize(size)
            hair_image = Image.composite(hair_image, self.mask_template, hair_mask_image)

            if hair_path is None:
                print('shape........', mask_path)

        seed = np.random.randint(2147483647)
        random.seed(seed)
        A = self.input_tf(A_img)
        random.seed(seed)
        B = self.input_tf(B_img)
        # random.seed(seed)
        # hair = self.input_tf(hair_image)
        # random.seed(seed)
        # hair_mask = self.mask_tf(hair_mask_image)

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
        # res = {'A': A, 'B': B, 
        #        'HairMask': hair_mask, 'Hair': hair,
        #        'AB_paths': AB_path}
        res = {'A': A, 'B': B, 
               'AB_paths': AB_path}
        return res

    def __len__(self):
        return (self.AB_size)

    def name(self):
        return 'ffhqSRDataset'
