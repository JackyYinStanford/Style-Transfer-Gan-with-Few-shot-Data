import torch
from .base_model import BaseModel
from . import networks
from .loss import PerceptualLoss
import cv2
from torch.autograd import Variable
import torch.nn.functional as F

class Pix2PixWarpGanModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='unaligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=50.0, help='weight for L1 loss')
            parser.add_argument('--lambda_GAN', type=float, default=0, help='weight for L1 loss')
            parser.add_argument('--lambda_vgg_loss', type=float, default=5000.0, help='weight for vgg loss')
            parser.add_argument('--lambda_mask_saturate', type=float, default=5.0, help='weight for mask_saturate')
            parser.add_argument('--lambda_mask_smooth', type=float, default=1e-4, help='weight for mask_smooth')
            parser.add_argument('--lambda_map', type=float, default=5000.0, help='weight for vgg loss between mapped image and original realB img')
            #parser.add_argument('--do_saturate_mask', type=bool, default=False,help='')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'map', 'Saturate', 'Smooth']
        if opt.use_vgg_loss:
            self.loss_names = ['G_GAN', 'G_L1', 'VGG_LOSS', 'D_real', 'D_fake', 'D_map', 'map']
        else:
            self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'D_map', 'map']
        self.perceptual_loss = PerceptualLoss()
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['fake_B', 'fake_B_mask', 'mapped_img', 'real_A', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G','D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.crop_size, opt.norm,
                                           not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                           with_mask=False, with_warp=True)
        if self.isTrain: 
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                               opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            
            self.criterionPerceptual = PerceptualLoss()
            if opt.use_vgg_loss:
                print('use vgg for realA to fakeB !!!!!!')
                self.criterionPerceptual = PerceptualLoss()
                
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_D)
            
        self.grid_size = opt.crop_size
        xx = torch.arange(0, self.grid_size).view(1,-1).repeat(self.grid_size,1)
        yy = torch.arange(0, self.grid_size).view(-1,1).repeat(1,self.grid_size)
        xx = xx.view(1,self.grid_size, self.grid_size)
        yy = yy.view(1,self.grid_size, self.grid_size)
        self.grid = torch.cat((xx,yy),0).float() # [2, H, W]

            
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['AB_paths' if AtoB else 'AB_paths']
        
    def set_input_for_test(self, input, path):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input.to(self.device)
        self.real_B = input.to(self.device)
        self.image_paths = path

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B, self.fake_B_mask, self.flow = self.netG(self.real_A)  # G(A) 返回的是生成图rgb，mask(alpha)以及optical flow(x, y)
        
        grid = self.grid.repeat(self.flow.shape[0],1,1,1)#[bs, 2, H, W]
        if self.real_A.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid, requires_grad = False) + self.flow
        vgrid = 2.0*vgrid/(self.grid_size-1)-1.0 #max(W-1,1)
        self.map_X = vgrid[:,0,:,:].unsqueeze(1)
        self.map_Y = vgrid[:,1,:,:].unsqueeze(1)
        vgrid = vgrid.permute(0,2,3,1)        
        mapped_img = F.grid_sample(self.real_A, vgrid)
        self.mapped_img = mapped_img
        # self.fake_B = mapped_img
        self.fake_B = (1 - self.fake_B_mask) * mapped_img + self.fake_B_mask * self.fake_B
        self.fake_B = torch.clamp(self.fake_B, -1.0, 1.0)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        
        #mapped_img; stop backprop to the generator by detaching fake_B
        map_AB = torch.cat((self.real_A, self.mapped_img), 1)
        pred_map = self.netD(map_AB.detach())
        self.loss_D_map = self.criterionGAN(pred_map, False)        
        
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real + self.loss_D_map) * (1.0/3)
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        # Second, G(A) = B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_map = self.criterionPerceptual(self.real_B, self.mapped_img) * self.opt.lambda_map
        self.loss_Saturate = torch.mean(self.fake_B_mask) * self.opt.lambda_mask_saturate
        self.loss_Smooth = self._compute_loss_smooth(self.fake_B_mask) * self.opt.lambda_mask_smooth
        # combine loss and calculate gradients
        if self.opt.use_vgg_loss:
            self.loss_VGG_LOSS = self.criterionPerceptual(self.fake_B, self.real_B) * self.opt.lambda_vgg_loss
            self.loss_G = self.loss_G_L1 + self.loss_VGG_LOSS + self.loss_G_GAN + self.loss_map
        else:
            self.loss_G = self.loss_G_L1 + self.loss_G_GAN + self.loss_map
            
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G        
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        
    def _do_if_necessary_saturate_mask(self, m, saturate=False):
        return torch.clamp(0.55*torch.tanh(3*(m-0.5))+0.5, 0, 1) if saturate else m

    def _compute_loss_smooth(self, mat):
        return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
            torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))

