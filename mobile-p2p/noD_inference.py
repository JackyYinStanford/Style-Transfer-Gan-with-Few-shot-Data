"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import glob
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
#from util import html
from util.util import tensor2im
from PIL import Image
import numpy as np
from data.base_dataset import BaseDataset, get_params, get_transform #get_cv_transform 
from torch.autograd import Variable
import torch
import cv2
import torchvision.transforms as transforms

mean = [0.5, 0.5, 0.5] 
std = [0.5, 0.5, 0.5] 
pil_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

def trans_data(path,opt):
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img[120-60:120+270,102+10:102+265-10,:]
    img = cv2.resize(img, (256, 256)) 
    img = Image.fromarray(np.uint8(img))
    w, h = img.size
    #transform_params = get_params(opt, (w,h))
    #transforms = get_transform(opt, transform_params, grayscale=False)
    img = pil_transforms(img)        
    return img
                
if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    
    # data under xxx
    img_paths = glob.glob('/data/tests/crop_video_frames/male/*/*')
    model.eval()
    
    for i, path in enumerate(img_paths):
        # if not ('test_male2' in path):
        #     continue
        
#         if not ('fcf' in path or 'test_male' in path or 'xiankun' in path):
#             continue
        if not ('fcf' in path):       # generate video
            continue
        save_path = path.replace('crop_video_frames','hairface_results/' + opt.name + '_gen_frame_pix2pix')
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
            
        data = trans_data(path, opt)
        print(path)
        tensor = Variable(torch.unsqueeze(data, dim=0).float(), requires_grad=False)

        model.set_input_for_test(tensor,path)  # unpack data from data loader
        img_path = model.get_image_paths()     # get image paths

#         (filepath, tempfilename) = os.path.split(img_path)
#         os.makedirs(filepath.replace('crop_video_frames', 'hairface_results/' + opt.name + '_gen_frame_pix2pix'), exist_ok=True)

        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        
        real_A = tensor2im(visuals['real_A'])
        fake_face_B = tensor2im(visuals['fake_B'])
        # fake_hair_rgba_B = tensor2im(visuals['fake_hair_rgba_B'])

        img = cv2.imread(path)
        img_gen = img.copy()
        fake_face_B = np.asarray(fake_face_B)
        #cv2.imwrite(save_path+'_face_.png', fake_face_B ,[cv2.IMWRITE_PNG_COMPRESSION,0])
        fake_face_B = cv2.resize(fake_face_B,(245,330))
        fake_face_B = cv2.cvtColor(fake_face_B, cv2.COLOR_RGB2BGR)
        img_gen[120-60:120+270,102+10:102+265-10,:] = fake_face_B
        cv2.imwrite(save_path.split('.')[0]+'_face_.png', img_gen ,[cv2.IMWRITE_PNG_COMPRESSION,0])
        
        # fake_hair_rgba_B = np.asarray(fake_hair_rgba_B)
        # fake_hair_rgba_B = cv2.resize(fake_hair_rgba_B,(448,448))
        # cv2.imwrite(save_path.split('.')[0]+'_hair_.png', fake_hair_rgba_B ,[cv2.IMWRITE_PNG_COMPRESSION,0])
        if i % 20 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))

