import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision
from math import exp
from torch.autograd import Variable

class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss



class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights
        print(' Using VGG19 for perceptualLoss... ')
    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])
        return content_loss



class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        vgg19 = models.vgg19(pretrained=False)
        vgg19.load_state_dict(torch.load('/data/pytorch-CycleGAN-and-pix2pix-master/pretrained_models/vgg19-dcbb9e9d.pth'))
        features = vgg19.features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out
    
# gram matrix and loss
class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        F = input.view(b, c, h * w)
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(h * w)
        return G


class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), GramMatrix()(target))
        return (out)


# vgg definition that conveniently let's you grab the outputs from any layer
class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]


class VGGLoss(nn.Module):
    def __init__(self, vgg=VGG()):
        super().__init__()
        self.vgg = vgg
        self.vgg.load_state_dict(torch.load('/home/jackyyin/Desktop/vgg16-397923af.pth'))
        for param in self.vgg.parameters():
            param.requires_grad = False
        if torch.cuda.is_available():
            self.vgg.cuda()
        self.style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
        self.style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
        self.l1 = nn.L1Loss()
        self.gram_loss = GramMSELoss()
        self.weight = 10.0


    def forward(self, input, target):
        feature_input = self.vgg(input, self.style_layers)
        feature_target = self.vgg(target, self.style_layers)
        loss_dict = {}
        loss = 0
        for i in range(5):
            loss += self.gram_loss(feature_input[i], feature_target[i])*self.style_weights[i]

        # loss2 = 0
        # for i in range(5):
        #     loss2 += self.l1(feature_input[i], feature_target[i])*self.style_weights[i]

        loss_dict['style'] = loss*self.weight
        loss_dict['l1'] = self.l1(input, target)
        return loss_dict


class VGGLossTotal(nn.Module):
    def __init__(self, vgg=VGG()):
        super().__init__()
        self.vgg = vgg
        self.vgg.load_state_dict(torch.load('/home/jackyyin/Desktop/vgg16-397923af.pth'))
        for param in self.vgg.parameters():
            param.requires_grad = False
        if torch.cuda.is_available():
            self.vgg.cuda()
        self.style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
        self.style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
        self.l1 = nn.L1Loss()
        self.gram_loss = GramMSELoss()
        self.weight = 10.0


    def forward(self, input, target):
        feature_input = self.vgg(input, self.style_layers)
        feature_target = self.vgg(target, self.style_layers)
        loss_dict = {}
        loss = 0
        for i in range(5):
            loss += self.gram_loss(feature_input[i], feature_target[i])*self.style_weights[i]

        # loss2 = 0
        # for i in range(5):
        #     loss2 += self.l1(feature_input[i], feature_target[i])*self.style_weights[i]
        loss_total = loss + self.l1(input, target)*self.weight
        return loss_total


def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
           torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        #vgg16 = models.vgg16(pretrained=True)        
        
        vgg16 = models.vgg16(pretrained=False)
        state_dict = torch.load('/home/jackyyin/Desktop/vgg16-397923af.pth')
        vgg16.load_state_dict(state_dict)
    
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


class DenoiseTVLoss(nn.Module):
    def __init__(self, extractor=VGG16FeatureExtractor()):
        super(DenoiseTVLoss, self).__init__()
        self.l1 = nn.MSELoss()
        self.extractor = extractor

    def forward(self, x, ground_truth):
        loss_dict = {}
        loss_dict['tv'] = total_variation_loss(x)
        loss_dict['content'] = self.l1(x, ground_truth)
        return loss_dict


class InpaintingLoss(nn.Module):
    def __init__(self, extractor=VGG16FeatureExtractor()):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor
        # self.conv = nn.Conv2d(3, 3, 3, stride=1, padding=1, bias=False)
        # torch.nn.init.constant_(self.conv.weight, 1.0)
        # for param in self.conv.parameters():
        #     param.requires_grad = False

    def forward(self, input, mask, output, ground_truth, ground_mask=None):
        loss_dict = {}
        output_comp = mask * input + (1 - mask) * output

        if ground_mask is None:
            loss_dict['hole'] = self.l1((1 - mask) * output, (1 - mask) * ground_truth)
            loss_dict['valid'] = self.l1(mask * output, mask * ground_truth)
        else:
            loss_dict['hole'] = self.l1((1 - ground_mask) * output, (1 - ground_mask) * ground_truth)
            loss_dict['valid'] = self.l1(ground_mask * output, ground_mask * ground_truth)

        feat_output_comp = self.extractor(output_comp)
        feat_output = self.extractor(output)
        feat_ground_truth = self.extractor(ground_truth)

        loss_dict['prc'] = 0.0
        for i in range(3):
            loss_dict['prc'] += self.l1(feat_output[i], feat_ground_truth[i])
            loss_dict['prc'] += self.l1(feat_output_comp[i], feat_ground_truth[i])

        loss_dict['style'] = 0.0
        for i in range(3):
            loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_ground_truth[i]))
            loss_dict['style'] += self.l1(gram_matrix(feat_output_comp[i]),
                                          gram_matrix(feat_ground_truth[i]))

        # with torch.no_grad():
        #     dilated_mask = self.conv(1-mask)
        #     no_update_holes = dilated_mask == 0
        #     new_mask = torch.ones_like(dilated_mask)
        #     new_mask = new_mask.masked_fill_(no_update_holes, 0.0)
        #     new_comp = output_comp * new_mask
        #
        # loss_dict['tv'] = total_variation_loss(new_comp) #(output_comp)

        # loss_dict['tv'] = self.l1(output_comp[:, :, :, :-1], output_comp[:, :, :, 1:]) + \
        #    self.l1(output_comp[:, :, :-1, :], output_comp[:, :, 1:, :])

        loss_dict['tv'] = total_variation_loss(output_comp)

        return loss_dict


LAMBDA_DICT = {
    'valid': 1.0, 'hole': 6.0, 'tv': 0.1, 'prc': 0.05, 'style': 120.00}
    #'valid': 1.0, 'hole': 6.0, 'tv': 0.1, 'prc': 0.05, 'style': 120.0}


class InpaintingLossNomaskTotal(nn.Module):
    def __init__(self, extractor=VGG16FeatureExtractor()):
        super(InpaintingLossNomaskTotal, self).__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor

    def forward(self, output, ground_truth):
        loss_dict = {}

        loss_dict['hole'] = self.l1(output, ground_truth)
        loss_dict['valid'] = 0.0

        feat_output = self.extractor(output)
        feat_ground_truth = self.extractor(ground_truth)

        loss_dict['prc'] = 0.0
        for i in range(3):
            loss_dict['prc'] += self.l1(feat_output[i], feat_ground_truth[i])*2.0

        loss_dict['style'] = 0.0
        for i in range(3):
            loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_ground_truth[i]))*2.0

        loss_dict['tv'] = total_variation_loss(output)

        loss = 0.0
        for key, coef in LAMBDA_DICT.items():
            value = coef * loss_dict[key]
            loss += value

        return loss


class InpaintingLossTotal(nn.Module):
    def __init__(self, extractor=VGG16FeatureExtractor()):
        super(InpaintingLossTotal, self).__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor
        # self.conv = nn.Conv2d(3, 3, 3, stride=1, padding=1, bias=False)
        # torch.nn.init.constant_(self.conv.weight, 1.0)
        # for param in self.conv.parameters():
        #     param.requires_grad = False

    def forward(self, input, mask, output, ground_truth, ground_mask=None):
        loss_dict = {}
        output_comp = mask * input + (1 - mask) * output

        if ground_mask is None:
            loss_dict['hole'] = self.l1((1 - mask) * output, (1 - mask) * ground_truth)
            loss_dict['valid'] = self.l1(mask * output, mask * ground_truth)
        else:
            loss_dict['hole'] = self.l1((1 - ground_mask) * output, (1 - ground_mask) * ground_truth)
            loss_dict['valid'] = self.l1(ground_mask * output, ground_mask * ground_truth)

        feat_output_comp = self.extractor(output_comp)
        feat_output = self.extractor(output)
        feat_ground_truth = self.extractor(ground_truth)

        loss_dict['prc'] = 0.0
        for i in range(3):
            loss_dict['prc'] += self.l1(feat_output[i], feat_ground_truth[i])
            loss_dict['prc'] += self.l1(feat_output_comp[i], feat_ground_truth[i])

        loss_dict['style'] = 0.0
        for i in range(3):
            loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_ground_truth[i]))
            loss_dict['style'] += self.l1(gram_matrix(feat_output_comp[i]),
                                          gram_matrix(feat_ground_truth[i]))

        # with torch.no_grad():
        #     dilated_mask = self.conv(1-mask)
        #     no_update_holes = dilated_mask == 0
        #     new_mask = torch.ones_like(dilated_mask)
        #     new_mask = new_mask.masked_fill_(no_update_holes, 0.0)
        #     new_comp = output_comp * new_mask
        #
        # loss_dict['tv'] = total_variation_loss(new_comp) #(output_comp)

        # loss_dict['tv'] = self.l1(output_comp[:, :, :, :-1], output_comp[:, :, :, 1:]) + \
        #    self.l1(output_comp[:, :, :-1, :], output_comp[:, :, 1:, :])

        loss_dict['tv'] = total_variation_loss(output_comp)

        loss = 0.0
        for key, coef in LAMBDA_DICT.items():
            value = coef * loss_dict[key]
            loss += value

        return loss


class InpaintingLossTotalNormalized(nn.Module):
    def __init__(self, extractor=VGG16FeatureExtractor()):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor
        # self.conv = nn.Conv2d(3, 3, 3, stride=1, padding=1, bias=False)
        # torch.nn.init.constant_(self.conv.weight, 1.0)
        # for param in self.conv.parameters():
        #     param.requires_grad = False

    def forward(self, input, mask, output, ground_truth, ground_mask=None):
        loss_dict = {}
        output_comp = mask * input + (1 - mask) * output

        h, w = mask.shape[2], mask.shape[3]

        if ground_mask is None:
            loss_dict['hole'] = self.l1((1 - mask) * output, (1 - mask) * ground_truth)/(h*w)
            loss_dict['valid'] = self.l1(mask * output, mask * ground_truth)/(h*w)
        else:
            loss_dict['hole'] = self.l1((1 - ground_mask) * output, (1 - ground_mask) * ground_truth)/(h*w)
            loss_dict['valid'] = self.l1(ground_mask * output, ground_mask * gt)/(h*w)

        feat_output_comp = self.extractor(output_comp)
        feat_output = self.extractor(output)
        feat_gt = self.extractor(gt)

        loss_dict['prc'] = 0.0
        for i in range(3):
            fc, fh, fw = feat_gt[i].shape[1], feat_gt[i].shape[2], feat_gt[i].shape[3]
            loss_dict['prc'] += self.l1(feat_output[i], feat_gt[i])/(fc * fh * fw)
            loss_dict['prc'] += self.l1(feat_output_comp[i], feat_gt[i])/(fc * fh * fw)

        loss_dict['style'] = 0.0
        for i in range(3):
            loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_gt[i]))
            loss_dict['style'] += self.l1(gram_matrix(feat_output_comp[i]),
                                          gram_matrix(feat_gt[i]))

        # with torch.no_grad():
        #     dilated_mask = self.conv(1-mask)
        #     no_update_holes = dilated_mask == 0
        #     new_mask = torch.ones_like(dilated_mask)
        #     new_mask = new_mask.masked_fill_(no_update_holes, 0.0)
        #     new_comp = output_comp * new_mask
        #
        # loss_dict['tv'] = total_variation_loss(new_comp) #(output_comp)

        # loss_dict['tv'] = self.l1(output_comp[:, :, :, :-1], output_comp[:, :, :, 1:]) + \
        #    self.l1(output_comp[:, :, :-1, :], output_comp[:, :, 1:, :])

        loss_dict['tv'] = total_variation_loss(output_comp)

        loss = 0.0
        for key, coef in LAMBDA_DICT.items():
            value = coef * loss_dict[key]
            loss += value

        return loss


class PerLossTotal(nn.Module):
    def __init__(self, extractor=VGG16FeatureExtractor()):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor
        # self.conv = nn.Conv2d(3, 3, 3, stride=1, padding=1, bias=False)
        # torch.nn.init.constant_(self.conv.weight, 1.0)
        # for param in self.conv.parameters():
        #     param.requires_grad = False
        self.weight = 0.01

    def forward(self, input, target):
        loss_dict = {}

        feat_input = self.extractor(input)
        feat_gt = self.extractor(target)

        loss_dict['prc'] = 0.0
        for i in range(3):
            loss_dict['prc'] += self.l1(feat_input[i], feat_gt[i])
            loss_dict['prc'] += self.l1(feat_input[i], feat_gt[i])

        loss_dict['l1'] = self.l1(input, target)

        loss = loss_dict['l1'] + loss_dict['prc'] * self.weight

        return loss


class WaterMarkLoss(nn.Module):
    def __init__(self, vgg=VGG()):
        super().__init__()
        self.vgg = vgg
        self.vgg.load_state_dict(torch.load('/home/huamiao/.torch/models/vgg_conv.pth'))
        for param in self.vgg.parameters():
            param.requires_grad = False
        if torch.cuda.is_available():
            self.vgg.cuda()
        self.style_layers = ['r22']
        self.l1 = nn.MSELoss() #nn.L1Loss()
        self.gram_loss = GramMSELoss()
        self.weight = 10.0


    def forward(self, input, target):
        feature_input = self.vgg(input, self.style_layers)
        feature_target = self.vgg(target, self.style_layers)
        loss_dict = {}
        loss = 0
        for i in range(len(self.style_layers)):
            _, c, h, w = feature_input[i].shape
            loss += self.l1(feature_input[i], feature_target[i])/(float(c*h*w))

        # loss2 = 0
        # for i in range(5):
        #     loss2 += self.l1(feature_input[i], feature_target[i])*self.style_weights[i]

        loss_dict['per'] = loss
        loss_dict['image'] = self.l1(input, target)
        return loss_dict


class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


# Perceptual loss that uses a pretrained VGG network
class VGGPerLoss(nn.Module):
    def __init__(self, device):
        super(VGGPerLoss, self).__init__()
        self.vgg = VGG19().to(device)#.cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        return total_variation_loss(x)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return 1.0 - ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


class MSSSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return 1.0 - msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)


class MSSSIML1Loss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIML1Loss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.l1 = nn.L1Loss()
        self.weight = 0.84

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        ssim = 1.0 - msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)
        l1 = self.l1(img1, img2)
        return ssim * self.weight + l1 * (1. - self.weight)


class NoiseMapLoss(torch.nn.Module):
    def __init__(self, alpha=0.3):
        super(NoiseMapLoss, self).__init__()
        self.alpha = alpha
        self.criterion = nn.MSELoss()

    def forward(self, input, output):
        mask = input < output
        mask = torch.sqrt(torch.abs(self.alpha - mask.float()))
        return self.criterion(mask*input, mask*output)


class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = torch.nn.L1Loss()
        self.mse_loss = torch.nn.MSELoss()

    @staticmethod
    def normalize_batch(batch, div_factor=255.):
        # normalize using imagenet mean and std
        mean = batch.data.new(batch.data.size())
        std = batch.data.new(batch.data.size())
        mean[:, 0, :, :] = 0.485
        mean[:, 1, :, :] = 0.456
        mean[:, 2, :, :] = 0.406
        std[:, 0, :, :] = 0.229
        std[:, 1, :, :] = 0.224
        std[:, 2, :, :] = 0.225
        batch = torch.div(batch, div_factor)

        batch -= Variable(mean)
        batch = torch.div(batch, Variable(std))
        return batch

    def forward(self, x, y):
        x = self.normalize_batch(x)
        y = self.normalize_batch(y)
        return self.l1_loss(x, y)


def make_vgg16_layers(style_avg_pool = False):
    """
    make_vgg16_layers
    Return a custom vgg16 feature module with avg pooling
    """
    vgg16_cfg = [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
        512, 512, 512, 'M', 512, 512, 512, 'M'
    ]

    layers = []
    in_channels = 3
    for v in vgg16_cfg:
        if v == 'M':
            if style_avg_pool:
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG16Partial(nn.Module):
    def __init__(self, vgg_path='/home/jackyyin/Desktop/vgg16-397923af.pth', layer_num=3):
        """
        Init
        :param layer_num: number of layers
        """
        super().__init__()
        vgg_model = models.vgg16(pretrained=True)
        vgg_model.features = make_vgg16_layers()
        # vgg_model.load_state_dict(
        #     torch.load(vgg_path, map_location='cpu')
        # )
        vgg_pretrained_features = vgg_model.features

        assert layer_num > 0
        assert isinstance(layer_num, int)
        self.layer_num = layer_num

        self.slice1 = torch.nn.Sequential()
        for x in range(5):  # 4
            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        if self.layer_num > 1:
            self.slice2 = torch.nn.Sequential()
            for x in range(5, 10):  # (4, 9)
                self.slice2.add_module(str(x), vgg_pretrained_features[x])

        if self.layer_num > 2:
            self.slice3 = torch.nn.Sequential()
            for x in range(10, 17):  # (9, 16)
                self.slice3.add_module(str(x), vgg_pretrained_features[x])

        if self.layer_num > 3:
            self.slice4 = torch.nn.Sequential()
            for x in range(17, 24):  # (16, 23)
                self.slice4.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

    @staticmethod
    def normalize_batch(batch, div_factor=1.0):
        # normalize using imagenet mean and std
        mean = batch.data.new(batch.data.size())
        std = batch.data.new(batch.data.size())
        mean[:, 0, :, :] = 0.485
        mean[:, 1, :, :] = 0.456
        mean[:, 2, :, :] = 0.406
        std[:, 0, :, :] = 0.229
        std[:, 1, :, :] = 0.224
        std[:, 2, :, :] = 0.225
        batch = torch.div(batch, div_factor)

        batch -= Variable(mean)
        batch = torch.div(batch, Variable(std))
        return batch

    def forward(self, x):
        h = self.slice1(x)
        h1 = h

        output = []

        if self.layer_num == 1:
            output = [h1]
        elif self.layer_num == 2:
            h = self.slice2(h)
            h2 = h
            output = [h1, h2]
        elif self.layer_num == 3:
            h = self.slice2(h)
            h2 = h
            h = self.slice3(h)
            h3 = h
            output = [h1, h2, h3]
        elif self.layer_num >= 4:
            h = self.slice2(h)
            h2 = h
            h = self.slice3(h)
            h3 = h
            h = self.slice4(h)
            h4 = h
            output = [h1, h2, h3, h4]
        return output


# perceptual loss and (spatial) style loss
class VGG16PartialLoss(PerceptualLoss):
    """
    VGG16 perceptual loss
    """

    def __init__(self, l1_alpha=5.0, perceptual_alpha=0.05, style_alpha=120,
                 smooth_alpha=0, feat_num=3, vgg_path='/home/jackyyin/Desktop/vgg16-397923af.pth'):
        super().__init__()

        self.vgg16partial = VGG16Partial(vgg_path=vgg_path).eval()

        self.loss_fn = torch.nn.L1Loss(size_average=True)

        self.l1_weight = l1_alpha
        self.vgg_weight = perceptual_alpha
        self.style_weight = style_alpha
        self.regularize_weight = smooth_alpha

        self.dividor = 1
        self.feat_num = feat_num

    def forward(self, output0_, target0_):
        """
        assuming both output0 and target0 are in the range of [0, 1]
        """
        output0 = (output0_ + 1.0)/2.0
        target0 = (target0_ + 1.0)/2.0
        y = self.normalize_batch(target0, self.dividor)
        x = self.normalize_batch(output0, self.dividor)

        # L1 loss
        l1_loss = self.l1_weight * (torch.abs(x - y).mean())
        vgg_loss = 0
        style_loss = 0
        smooth_loss = 0

        # VGG
        if self.vgg_weight != 0 or self.style_weight != 0:

            yc = Variable(y.data)

            with torch.no_grad():
                groundtruth = self.vgg16partial(yc)
            generated = self.vgg16partial(x)

            # vgg loss: VGG content loss
            if self.vgg_weight > 0:
                # for m in range(0, len(generated)):
                for m in range(len(generated) - self.feat_num, len(generated)):
                    gt_data = Variable(groundtruth[m].data, requires_grad=False)
                    vgg_loss += (
                            self.vgg_weight * self.loss_fn(generated[m], gt_data)
                    )

            # style loss: Gram matrix loss
            if self.style_weight > 0:
                # for m in range(0, len(generated)):
                for m in range(len(generated) - self.feat_num, len(generated)):
                    gt_style = gram_matrix(
                        Variable(groundtruth[m].data, requires_grad=False))
                    gen_style = gram_matrix(generated[m])
                    style_loss += (
                            self.style_weight * self.loss_fn(gen_style, gt_style)
                    )

        # smooth term
        if self.regularize_weight != 0:
            smooth_loss += self.regularize_weight * (
                    torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]).mean() +
                    torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]).mean()
            )

        tot = l1_loss + vgg_loss + style_loss + smooth_loss
        return tot #, vgg_loss, style_loss


class InpaintPartialMixLoss(nn.Module):
    def __init__(self, mix_weight=0.5):
        super().__init__()

        self.inpaint_loss = InpaintingLossTotal()
        self.partial_loss = VGG16PartialLoss()
        self.mix_weight = mix_weight

    def forward(self, output, gt):
        mask = torch.zeros_like(gt)
        inpaint = self.inpaint_loss(gt, mask, output, gt)
        partial = self.partial_loss(output, gt)
        return inpaint * self.mix_weight + partial * (1. - self.mix_weight)


