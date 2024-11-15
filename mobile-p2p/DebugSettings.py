#!/usr/bin/env python
# -*- coding: utf-8 -*-

# torch==0.4.0
# torchvision==0.2.1
# dominate>=2.3.1
# visdom>=0.1.8.3

#pip install --user scipy==1.1.0 -i https://mirrors.tencent.com/pypi/simple/
import os
os.system('pip install pandas')
os.system('pip install opencv-python')
os.system('pip install opencv-contrib-python')
os.system('pip install scipy')
os.system('pip install scikit-image -i https://mirrors.tencent.com/pypi/simple/')
os.system('pip install torch==1.2.0')
os.system('pip install torchvision>=0.2.1')
os.system('pip install dominate>=2.3.1')
os.system('pip install torchsummary')
os.system('pip install opencv_transforms')

os.system('sh /data/pytorch-CycleGAN-and-pix2pix-master/train.sh \
--crop_size 256 \
--model pix2pix_attention_gan_hair_face \
--name test_hairgan_250_new_smoothlossToFace1_bigDebug \
--ngf 16 \
--ndf 64 \
--gan_mode vanilla \
--netD basic \
--dataset_mode aligned_hair_face \
--norm batch \
--batch_size 16 \
--gpu_ids 0 \
--netG mobileunetv3_2_0_0 \
--load_memory \
--lambda_L1 250.0 \
--lambda_mask_smooth 1 \
--lambda_face_gan 0.5 \
--dataroot /data/facecyclegan/pairs_for_distill/step7_2_crop5_male2beauty_stylegan_cyc50_vgg10 \
1>/data/log11.log 2>/data/err11.log')
