import os.path
# from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
from PIL import Image
import random
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader,Dataset
#from opencv_transforms import opencv_transforms as transforms
#from opencv_transforms import transforms 
import cv2


class SRDataset(Dataset):
    def __init__(self, dataroot):
#         self.dir_AB = os.path.join(dataroot)  # get the image directory
        self.AB_paths = [dataroot + nums for nums in os.listdir(dataroot)]
        self.AB_size = len(self.AB_paths)

        self.mean = [0.5, 0.5, 0.5] #[0.485, 0.456, 0.406]
        self.std = [0.5, 0.5, 0.5] #[0.229, 0.224, 0.225]
        osize = [256, 256]
        self.AB_len = len(self.AB_paths)

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
                #transforms.RandomHorizontalFlip(),
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
        self.A_images = [1]*self.AB_len
        self.B_images = [1]*self.AB_len
        import tqdm
        for index in tqdm.tqdm(range(self.AB_len)):
#             AB_img = Image.open(self.AB_paths[index]).convert('RGB')
            AB_img = cv2.imread(self.AB_paths[index])
            AB_img = cv2.cvtColor(AB_img,cv2.COLOR_BGR2RGB)
            w, h = AB_img.size
            w2 = int(w / 2)
            self.A_images[index] = AB_img.crop((0, 0, w2, h))
            self.B_images[index] = AB_img.crop((w2, 0, w, h))

    def __getitem__(self, index):
        size = (256,256)

        # For Debug
        # index = index%1280
        if index == 0:
            seedseed = np.random.randint(2147483647)
            seed = np.random.randint(seedseed)
            random.seed(seed)
            random.shuffle(self.A_images)
            random.seed(seed)
            random.shuffle(self.B_images)

        A_img = self.A_images[index]
        B_img = self.B_images[index]
        AB_path = self.AB_paths[index]

        seedseed = np.random.randint(2147483647)
        seed = np.random.randint(seedseed)   
        random.seed(seed)
        A = self.input_tf(A_img)
        random.seed(seed)
        B = self.input_tf(B_img)

        res = {'A': A, 'B': B, 
               'AB_paths': AB_path}
        return res

    def __len__(self):
        return (self.AB_size)

    def name(self):
        return 'SRDataset'
