'''
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

def make_model(args, parent=False):
    return EndoSR(args)



def MyConv2d(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='elu', mode='CNA'):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)


class EMHA(nn.Module):
    def __init__(self, inChannels, splitfactors=8, heads=4):
        super().__init__()
        dimHead = inChannels // (2*heads)

        self.heads = heads
        self.splitfactors = splitfactors
        self.scale = dimHead ** -0.5

        self.reduction = nn.Conv1d(
            in_channels=inChannels, out_channels=inChannels//2, kernel_size=1)
        self.attend = nn.Softmax(dim=-1)
        self.toQKV = nn.Linear(
            inChannels // 2, inChannels // 2 * 3, bias=False)
        self.expansion = nn.Conv1d(
            in_channels=inChannels//2, out_channels=inChannels, kernel_size=1)

    def forward(self, x):
        x = self.reduction(x)
        x = x.transpose(-1, -2)

        qkv = self.toQKV(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        qs, ks, vs = map(lambda t: t.chunk(
            self.splitfactors, dim=2), [q, k, v])

        pool = []
        for qi, ki, vi in zip(qs, ks, vs):
            tmp = torch.matmul(qi, ki.transpose(-1, -2)) * self.scale
            attn = self.attend(tmp)
            out = torch.matmul(attn, vi)
            out = rearrange(out, 'b h n d -> b n (h d)')
            pool.append(out)

        out = torch.cat(tuple(pool), dim=1)
        out = out.transpose(-1, -2)
        out = self.expansion(out)
        return out

def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer

def act(act_type, inplace=True, neg_slope=0.2, n_selu=1):
    # helper selecting activation
    # neg_slope: for selu and init of selu
    # n_selu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU()
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(0.2,inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU()
    elif act_type == 'sigmoid':
        layer = nn.Sigmoid()
    elif act_type == 'selu':
        layer = nn.SELU()
    elif act_type == 'elu':
        layer = nn.ELU()
    elif act_type == 'silu':
        layer = nn.SiLU()
    elif act_type == 'rrelu':
        layer = nn.RReLU()
    elif act_type == 'celu':
        layer = nn.CELU()
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer

def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)



class BB(nn.Module) :
  def __init__(self, in_channels, out_channels, splitfactors=4, heads=8, k=3) :
    super(BB, self).__init__()
    self.k = k
    self.uk3_1 = MyConv2d(in_nc=in_channels, out_nc=in_channels, kernel_size= 3)
    self.uk3_2 = MyConv2d(in_nc=2*in_channels, out_nc=in_channels, kernel_size= 3)
    self.uk3_3 = MyConv2d(in_nc=3*in_channels, out_nc=in_channels, kernel_size= 3)
    self.lk3_1 = MyConv2d(in_nc=in_channels, out_nc=in_channels, kernel_size= 5)
    self.lk3_2 = MyConv2d(in_nc=2*in_channels, out_nc=in_channels, kernel_size= 5)
    self.lk3_3 = MyConv2d(in_nc=3*in_channels, out_nc=in_channels, kernel_size= 5)
    self.k1 = MyConv2d(in_nc=4*in_channels, out_nc=in_channels, kernel_size= 1)
    self.k2 = MyConv2d(in_nc=4*in_channels, out_nc=in_channels, kernel_size= 1)
    self.emha = EMHA(in_channels*k*k, splitfactors, heads)
    self.norm = nn.LayerNorm(in_channels*k*k)
    self.unFold = nn.Unfold(kernel_size=(k, k), padding=1)
    self.conv = MyConv2d(in_channels*4, out_channels,3)
  
  def forward(self,x):
    _, _, h, w = x.shape

    #upper path
    xu1_1= self.uk3_1(x)
    xu1_2= torch.cat((xu1_1,x),1)
    xu2_1= self.uk3_2(xu1_2)
    xu2_2= torch.cat((xu2_1,xu1_1,x),1)
    xu3_1= self.uk3_3(xu2_2)
    xu3_2= torch.cat((xu3_1,xu2_1,xu1_1,x),1)
    xu3= self.k1(xu3_2)
    #lower path
    xl1_1= self.lk3_1(x)
    xl1_2= torch.cat((xl1_1,x),1)
    xl2_1= self.lk3_2(xl1_2)
    xl2_2= torch.cat((xl2_1,xl1_1,x),1)
    xl3_1= self.lk3_3(xl2_2)
    xl3_2= torch.cat((xl3_1,xl2_1,xl1_1,x),1)
    xl3= self.k2(xl3_2)
    #transformer 

    xt1 = self.unFold(x)
    xt2 = xt1.transpose(-2, -1)
    xt2 = self.norm(xt2)
    xt2 = xt2.transpose(-2, -1)
    xt2 = self.emha(xt2)+xt1
    xt2 = F.fold(xt2, output_size=(h, w), kernel_size=(self.k, self.k), padding=(1, 1))
    xt2 = xt2+x

    #concatanating the blocks
    out = torch.cat((xu3,xl3,xt2,x), 1)
    out = self.conv(out)  
      
    return out
  

class Degradation_model(nn.Module):
    def __init__(self, in_channels, out_channels, n_feats=32, n_blocks=8, scale=2, bias=True):
        super(Degradation_model, self).__init__()
        self.head = nn.Sequential(nn.Conv2d(in_channels, n_feats, kernel_size=3, stride=1, padding=1), nn.ReLU(True))
        body = [ResBlock_D(in_channels=n_feats, out_channels=n_feats, bias=bias) for _ in range(n_blocks)]
        body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1))
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(nn.Conv2d(n_feats, out_channels, kernel_size=scale, stride=scale))

    def forward(self, x):
        x = self.head(x)
        x = self.body(x) + x
        x = self.tail(x)

        return x


class ResBlock_D(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(ResBlock_D, self).__init__()
        self.Block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(True),
                                   nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        x = self.Block(x) + x
        return x


# class ResBlock_R(nn.Module):
#     def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, res_scale=1, bias=True):
#         super(ResBlock_R, self).__init__()
#         self.res_scale = res_scale
#         layer = []
#         layer.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
#         layer.append(nn.ReLU(True))
#         layer.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
#         self.res = nn.Sequential(*layer)

#     def forward(self, x):
#         return x + self.res(x) * self.res_scale



# class Encode_model(nn.Module):
#     def __init__(self, in_channels, n_blocks=8, scale=2):
#         super(Encode_model, self).__init__()
#         if scale == 2:
#             res_scale = 1
#             n_feats = 64
#         elif scale == 4:
#             res_scale = 0.1
#             n_feats = 128
#         elif scale == 8:
#             res_scale = 0.1
#             n_feats = 128
#         self.head = nn.Sequential(nn.Conv2d(in_channels, n_feats, kernel_size=3, stride=1, padding=1), nn.ReLU(True))
#         body = [ResBlock_R(in_channels=n_feats, out_channels=n_feats, res_scale=res_scale) for _ in range(n_blocks)]
#         body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1))
#         self.body = nn.Sequential(*body)

#     def forward(self, x):
#         x = self.head(x)
#         x = self.body(x)

#         return x


# class Reconstruction_model(nn.Module):
#     def __init__(self, out_channels, n_blocks=8, scale=2):
#         super(Reconstruction_model, self).__init__()
#         if scale == 2:
#             res_scale = 1
#             n_feats = 64
#             self.tail = nn.Sequential(nn.Conv2d(n_feats, n_feats*4, kernel_size=3, stride=1, padding=1),
#                                       nn.PixelShuffle(2),
#                                       nn.Conv2d(n_feats, out_channels, kernel_size=3, stride=1, padding=1))
#         elif scale == 4:
#             res_scale = 0.1
#             n_feats = 128
#             self.tail = nn.Sequential(nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, stride=1, padding=1),
#                                       nn.PixelShuffle(2),
#                                       nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, stride=1, padding=1),
#                                       nn.PixelShuffle(2),
#                                       nn.Conv2d(n_feats, out_channels, kernel_size=3, stride=1, padding=1)
#                                       )
#         elif scale == 8:
#             res_scale = 0.1
#             n_feats = 128
#             self.tail = nn.Sequential(nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, stride=1, padding=1),
#                                       nn.PixelShuffle(2),
#                                       nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, stride=1, padding=1),
#                                       nn.PixelShuffle(2),
#                                       nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, stride=1, padding=1),
#                                       nn.PixelShuffle(2),
#                                       nn.Conv2d(n_feats, out_channels, kernel_size=3, stride=1, padding=1)
#                                       )
#         # self.head = nn.Sequential(nn.Conv2d(in_channels, n_feats, kernel_size=3, stride=1, padding=1), nn.ReLU(True))
#         body = [ResBlock_R(in_channels=n_feats, out_channels=n_feats, res_scale=res_scale) for _ in range(n_blocks)]
#         body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1))
#         self.body = nn.Sequential(*body)

#     def forward(self, x):
#         # x = self.head(x)
#         x = self.body(x)
#         x = self.tail(x)
#         return x
    
class ResBlock_R (nn.Module) :
    def __init__(self, in_channels, out_channels, n_bb=3, splitfactors=8, heads=4, k=3) :
        super(ResBlock_R, self).__init__()
        self.body = nn.ModuleList(
            [BB(in_channels=in_channels, out_channels=out_channels, splitfactors=splitfactors, heads=heads, k=k) for _ in range(n_bb)])

    def forward(self, x):
        for bb in self.body:
            x = x + bb(x)
        return x
 

      
class Encode_model(nn.Module):
    def __init__(self, in_channels, n_blocks=5, scale=4):
        super(Encode_model, self).__init__()
        if scale == 2:
            res_scale = 1
            n_feats = 64
        elif scale == 4:
            res_scale = 0.1
            n_feats = 128
        elif scale == 8:
            res_scale = 0.1
            n_feats = 128
        self.head = nn.Sequential(nn.Conv2d(in_channels, n_feats, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(True))
        body = [ResBlock_R(in_channels=n_feats, out_channels=n_feats, n_bb=3, splitfactors=4, heads=4, k=3) for _ in range(n_blocks)]
        body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1))
        self.body = nn.Sequential(*body)
        self.conv = MyConv2d(in_nc=in_channels, out_nc=n_feats, kernel_size=1)

    def forward(self, x):
        out1 = self.head(x)
        out = self.body(out1)
        return out + out1


class Reconstruction_model(nn.Module):
    def __init__(self, out_channels, n_blocks=5, scale=4):
        super(Reconstruction_model, self).__init__()
        if scale == 2:
            res_scale = 1
            n_feats = 64
            self.tail = nn.Sequential(nn.Conv2d(n_feats, n_feats*4, kernel_size=3, stride=1, padding=1),
                                      nn.PixelShuffle(2),
                                      nn.Conv2d(n_feats, out_channels, kernel_size=3, stride=1, padding=1))
        elif scale == 4:
            res_scale = 0.1
            n_feats = 128
            self.tail = nn.Sequential(nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, stride=1, padding=1),
                                      nn.PixelShuffle(2),
                                      nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, stride=1, padding=1),
                                      nn.PixelShuffle(2),
                                      nn.Conv2d(n_feats, out_channels, kernel_size=3, stride=1, padding=1)
                                      )
        elif scale == 8:
            res_scale = 0.1
            n_feats = 128
            self.tail = nn.Sequential(nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, stride=1, padding=1),
                                      nn.PixelShuffle(2),
                                      nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, stride=1, padding=1),
                                      nn.PixelShuffle(2),
                                      nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, stride=1, padding=1),
                                      nn.PixelShuffle(2),
                                      nn.Conv2d(n_feats, out_channels, kernel_size=3, stride=1, padding=1)
                                      )
        # self.head = nn.Sequential(nn.Conv2d(in_channels, n_feats, kernel_size=3, stride=1, padding=1), nn.ReLU(True))
        body = [ResBlock_R(in_channels=n_feats, out_channels=n_feats) for _ in range(n_blocks)]
        body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        # x = self.head(x)
        out1 = self.body(x)
        out = self.tail(out1+x)
        return out


class EndoSR(nn.Module):
    def __init__(self, args):
        super(EndoSR, self).__init__()
        self.scale = int(args.scale)
        if self.scale == 2:
            res_scale = 1
            n_feats = 64
        elif self.scale == 4:
            res_scale = 0.1
            n_feats = 128
        elif self.scale == 8:
            res_scale = 0.1
            n_feats = 128
        self.encode = Encode_model(in_channels=3, n_blocks=5, scale=self.scale)
        self.recon = Reconstruction_model(out_channels=3, n_blocks=5, scale=self.scale)

    def forward(self, x):
        feature = self.encode(x)
        img = self.recon(feature)
        return feature, img


# Defines the PatchGAN discriminator with the specified arguments.
class Discriminator_model(nn.Module):
    def __init__(self, scale, n_layers=3, norm_layer=nn.InstanceNorm2d, use_sigmoid=False):
        super(Discriminator_model, self).__init__()
        if scale == 2:
            ndf = 64
        elif scale == 4:
            ndf = 128
        elif scale == 8:
            ndf = 128

        use_bias = False
        kw = 4
        padw = 1
        sequence = []

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                         norm_layer(ndf * nf_mult),
                         nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = 2
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                     norm_layer(ndf * nf_mult),
                     nn.LeakyReLU(0.2, True)]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)