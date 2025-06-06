import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
from models.fpn_mobilenet import FPNMobileNet

#############################################

from models.fpn_ghostnet import FPNGhostNet, HINet

#############################################

###############################################################################
# Functions
###############################################################################
torch.cuda.set_device(0)

def get_norm_layer(norm_type='instance', affine= False):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=affine, track_running_stats=True)
    elif norm_type=='hin':
        norm_layer= HINet
        
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

##############################################################################
# Classes
##############################################################################


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, use_parallel=True, learn_residual=True, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        output = self.model(input)
        if self.learn_residual:
            output = input + output
            output = torch.clamp(output,min = -1,max = 1)
        return output


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class DicsriminatorTail(nn.Module):
    def __init__(self, nf_mult, n_layers, ndf=64, norm_layer=nn.BatchNorm2d, use_parallel=True):
        super(DicsriminatorTail, self).__init__()
        self.use_parallel = use_parallel
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw-1)/2))

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence = [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, norm_layer=nn.BatchNorm2d, use_parallel=True):
        super(MultiScaleDiscriminator, self).__init__()
        self.use_parallel = use_parallel
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, 3):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        self.scale_one = nn.Sequential(*sequence)
        self.first_tail = DicsriminatorTail(nf_mult=nf_mult, n_layers=3)
        nf_mult_prev = 4
        nf_mult = 8

        self.scale_two = nn.Sequential(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True))
        nf_mult_prev = nf_mult
        self.second_tail = DicsriminatorTail(nf_mult=nf_mult, n_layers=4)
        self.scale_three = nn.Sequential(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True))
        self.third_tail = DicsriminatorTail(nf_mult=nf_mult, n_layers=5)

    def forward(self, input):
        x = self.scale_one(input)
        x_1 = self.first_tail(x)
        x = self.scale_two(x)
        x_2 = self.second_tail(x)
        x = self.scale_three(x)
        x = self.third_tail(x)
        return [x_1, x_2, x]


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, use_parallel=True):
        super(NLayerDiscriminator, self).__init__()
        self.use_parallel = use_parallel
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


def get_fullD(model_config):
    model_d = NLayerDiscriminator(n_layers=5,
                                  norm_layer=get_norm_layer(norm_type=model_config['norm_layer']),
                                  use_sigmoid=False)
    return model_d



# def get_generator(model_config, cuda= True):
    # generator_name = model_config['g_name']
def get_generator(model_config, cuda=True):
    # 支持直接传入字符串模型名称
    default_config = {
        'norm_layer': 'hin',  # 默认使用hin
        'learn_residual': True,
        'dropout': True,
        'blocks': 9
    }

    if isinstance(model_config, str):
        generator_name = model_config
        model_config = default_config  # 使用默认配置
    else:
        generator_name = model_config['g_name']

    if generator_name == 'fpn_inception':
        model_g = FPNInception(norm_layer=get_norm_layer(norm_type=model_config.get('norm_layer', 'instance')))
    elif generator_name == 'fpn_mobilenet':
        model_g = FPNMobileNet(norm_layer=get_norm_layer(norm_type=model_config.get('norm_layer', 'instance')))
    elif generator_name == 'fpn_inception_simple':
        model_g = FPNInceptionSimple()
    elif generator_name == 'fpn_densenet':
        model_g = FPNDenseNet()
    elif generator_name == 'fpn_mobilenet_v2':
        model_g = FPNMobileNetV2(norm_layer=get_norm_layer(norm_type=model_config.get('norm_layer', 'instance')))
    elif generator_name == 'fpn_ghostnet':
        model_g = FPNGhostNet(norm_layer=get_norm_layer(norm_type=model_config.get('norm_layer', 'instance')))
    elif generator_name == 'fpn_ghostnet_gm':
        model_g = FPNGhostNet(norm_layer=get_norm_layer(norm_type=model_config.get('norm_layer', 'instance')))
    elif generator_name == 'fpn_ghostnet_gm_hin':
        model_g = FPNGhostNet(norm_layer=get_norm_layer(norm_type=model_config.get('norm_layer', 'hin'), affine=True))

    #elif generator_name == 'fpn_ghostnet_gm':
     #   model_g = FPNGhostNet_GM(norm_layer=get_norm_layer(norm_type=model_config.get('norm_layer', 'instance')))
    #elif generator_name == 'fpn_ghostnet_gm_hin':
     #   model_g = FPNGhostNet_GM(
      #      norm_layer=get_norm_layer(norm_type=model_config.get('norm_layer', 'hin'), affine=True))


    elif generator_name == 'resnet_9blocks':
        model_g = ResnetGenerator(norm_layer=get_norm_layer(norm_type=model_config.get('norm_layer', 'instance')),
                                  use_dropout=model_config.get('dropout', True), n_blocks=model_config.get('blocks', 9),
                                  learn_residual=model_config.get('learn_residual', True))
    elif generator_name == 'unet_128':
        model_g = UnetGenerator(norm_layer=get_norm_layer(norm_type=model_config.get('norm_layer', 'instance')),
                                use_dropout=model_config.get('dropout', True))
    elif generator_name == 'unet_256':
        model_g = UnetGenerator(norm_layer=get_norm_layer(norm_type=model_config.get('norm_layer', 'instance')),
                                use_dropout=model_config.get('dropout', True))
    else:
        raise ValueError("Generator Network [%s] not recognized." % generator_name)

    model_g = nn.DataParallel(model_g)
    if cuda:
        model_g = model_g.cuda()

    return model_g


def get_discriminator(model_config):
    discriminator_name = model_config['d_name']
    if discriminator_name == 'no_gan':
        model_d = None
    elif discriminator_name == 'patch_gan':
        model_d = NLayerDiscriminator(n_layers=model_config['d_layers'],
                                      norm_layer=get_norm_layer(norm_type=model_config['norm_layer']),
                                      use_sigmoid=False)
        model_d = nn.DataParallel(model_d, device_ids= [0])
    elif discriminator_name == 'double_gan':
        patch_gan = NLayerDiscriminator(n_layers=model_config['d_layers'],
                                        norm_layer=get_norm_layer(norm_type=model_config['norm_layer']),
                                        use_sigmoid=False)
        patch_gan = nn.DataParallel(patch_gan, device_ids= [0])
        full_gan = get_fullD(model_config)
        full_gan = nn.DataParallel(full_gan, device_ids= [0])
        model_d = {'patch': patch_gan,
                   'full': full_gan}
    elif discriminator_name == 'multi_scale':
        model_d = MultiScaleDiscriminator(norm_layer=get_norm_layer(norm_type=model_config['norm_layer']))
        model_d = nn.DataParallel(model_d, device_ids= [0])
    else:
        raise ValueError("Discriminator Network [%s] not recognized." % discriminator_name)

    return model_d


def get_nets(model_config):
    return get_generator(model_config), get_discriminator(model_config)
