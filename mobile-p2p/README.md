# Prerequisite
1. Install relevant libraries according to requirements.txt
2. Download vgg16-397923af.pth from 'https://download.pytorch.org/models/vgg16-397923af.pth' and revise the path in current folder to your download path. It's for the vgg loss.

# Structure
1. Model relevant codes are in models folder.
2. Configuration relevant codes are in options folder.
3. Data preprocessing codes are in data folder.

# Training commands

```shell
sh train.sh \
--crop_size 320 \
--model pix2pix_attention_gan \
--name asian_comics \
--ngf 20 \
--ndf 64 \
--gan_mode vanilla \
--netD basic \
--dataset_mode aligned_attention \
--norm batch \
--batch_size 240 \
--gpu_ids 0 \
--netG mobileunetv3_2_0_0 \
--load_memory \
--lambda_L1 250.0 \
--lambda_mask_smooth 0 \
--dataroot /localhome/local-jackyyin/generated_dataset/us_comcis/

# if you have more gpus, it will accelerate the training speed much faster, please revise the --gpu_ids param in above commands.
```
if you have more gpus, it will accelerate the training speed much faster, please revise the --gpu_ids param in above commands.

# Visualize the training results
tensorboard --logdir=event/asain_comics (change to your event save dir)
