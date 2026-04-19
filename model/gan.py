'''
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

def make_model(args, parent=False):
    return Generator_model(args)



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
    def __init__(self, inChannels, splitfactors=4, heads=1):
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
    elif act_type == 'lrelu' or 'leakyrelu':
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
    elif act_type == 'sigm':
        layer = nn.Sigmoid()
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

class SimpleAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.scale = channels ** -0.5
        self.conv_q = nn.Conv2d(channels, channels, 1)
        self.conv_k = nn.Conv2d(channels, channels, 1)
        self.conv_v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.InstanceNorm2d(channels)
        
    def forward(self, x):
        B, C, H, W = x.shape
        q = self.conv_q(x).view(B, C, -1)
        k = self.conv_k(x).view(B, C, -1)
        v = self.conv_v(x).view(B, C, -1)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (v @ attn).view(B, C, H, W)
        out = self.proj(out)
        out = self.norm(out)
        return out + x

class SimplifiedBB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Main convolution branch
        self.conv1 = MyConv2d(in_channels, in_channels, 3,act_type='elu',norm_type='instance',groups=16)
        self.conv2 = MyConv2d(in_channels, in_channels, 3,act_type='elu',norm_type='instance',groups=16)
        self.conv3 = MyConv2d(in_channels, in_channels, 3,act_type='elu',norm_type='instance',groups=16)
        self.conv4 = MyConv2d(in_channels, in_channels, 3,act_type='elu',norm_type='instance',groups=16)
        
        # Simple attention branch
        self.attention = SimpleAttention(in_channels)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1)
        )
        
        # Scaling factors for stability
        self.conv_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.attn_scale = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x):
        # Convolution path
        conv_out1 = self.conv1(x)
        conv_out2 = self.conv2(x)
        conv_out3 = self.conv3(x)
        conv_out4 = self.conv4(x) * self.conv_scale

        
        # Attention path
        attn_out = self.attention(x) * self.attn_scale
        
        # Combine paths
        combined = torch.cat([conv_out4, attn_out], dim=1)
        out = self.fusion(combined)
        
        return out + x

class EnhancedBB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1st convolution branch
        self.conv3_1 = MyConv2d(in_channels, in_channels, 3,act_type='elu',norm_type='instance',groups=16)
        self.conv3_2 = MyConv2d(in_channels, in_channels, 3,act_type='elu',norm_type='instance',groups=16)
        self.conv3_3 = MyConv2d(in_channels, in_channels, 3,act_type='elu',norm_type='instance',groups=16)

        # 2nd branch  
        self.conv5_1 = MyConv2d(in_channels, in_channels, 5,act_type='elu',norm_type='instance',groups=16)
        self.conv5_2 = MyConv2d(in_channels, in_channels, 5,act_type='elu',norm_type='instance',groups=16)
        self.conv5_3 = MyConv2d(in_channels, in_channels, 5,act_type='elu',norm_type='instance',groups=16)
        
        # Simple attention branch
        self.attention = AttentionLayer(in_channels)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, 1),
            nn.InstanceNorm2d(out_channels)
        )
        
        # Scaling factors for stability
        self.conv_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.attn_scale = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x):
        # Convolution path
        conv_out3_1 = self.conv3_1(x)
        conv_out3_2 = self.conv3_2(conv_out3_1)
        conv_out3_3 = self.conv3_3(conv_out3_2) * self.conv_scale

        conv_out5_1 = self.conv5_1(x)
        conv_out5_2 = self.conv5_2(conv_out5_1)
        conv_out5_3 = self.conv5_3(conv_out5_2) * self.conv_scale

        
        # Attention path
        attn_out = self.attention(x) * self.attn_scale
        
        # Combine paths
        combined = torch.cat([conv_out3_3, conv_out5_3, attn_out], dim=1)
        out = self.fusion(combined)
        
        return out + x

class AttentionLayer(nn.Module):
    def __init__(self, n_feats):
        super(AttentionLayer, self).__init__()
        self.esa = ESA(n_feats, nn.Conv2d)  
        self.simple_attention = SimpleAttention(n_feats)
        self.ln1 = nn.LayerNorm(n_feats)  

    def forward(self, x):
        B, C, H, W = x.shape
        
        x_ln = x.permute(0, 2, 3, 1).contiguous()  # Reshape to (B, H, W, C)
        x_ln = self.ln1(x_ln).permute(0, 3, 1, 2).contiguous()  # Back to (B, C, H, W)

        attn_out1 = self.esa(x_ln)
        attn_out2 = self.simple_attention(x_ln)

        out = attn_out1 + attn_out2 + x
        # out = attn_out2 + x

        return out


class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bicubic', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m

class BB(nn.Module):
    def __init__(self, in_channels, out_channels, splitfactors=4, heads=8, k=3):
        super(BB, self).__init__()
        self.k = k
        self.patch_size = k  # Kernel size for unfolding/folding
        self.padding = 1     # Padding for unfolding/folding
        
        # Layers
        self.unFold = nn.Unfold(kernel_size=(k, k), padding=self.padding)
        self.fold = nn.Fold(output_size=None, kernel_size=(k, k), padding=self.padding)  # Remove output_size
        self.norm = nn.LayerNorm(in_channels * k * k)
        self.emha1 = EMHA(in_channels * k * k, splitfactors, heads)
        self.emha2 = EMHA(in_channels * k * k, splitfactors, heads)
        self.emha3 = EMHA(in_channels * k * k, splitfactors, heads)
        self.conv = MyConv2d(in_channels, out_channels, 3,act_type='elu')

    def forward(self, x):
        _, _, h, w = x.shape
        # Compute output size for folding
        fold_output_size = (h, w)

        # First Transformer Block
        xt1 = self.unFold(x)
        xt1_norm = self.norm(xt1.transpose(-2, -1)).transpose(-2, -1)
        xt1_transformed = self.emha1(xt1_norm) + xt1
        xt1_folded = F.fold(xt1_transformed, output_size=(h, w), kernel_size=(self.k, self.k), padding=(1, 1))

        # Second Transformer Block
        xt2 = self.unFold(xt1_folded)
        xt2_norm = self.norm(xt2.transpose(-2, -1)).transpose(-2, -1)
        xt2_transformed = self.emha2(xt2_norm) + xt2
        xt2_folded = F.fold(xt2_transformed, output_size=(h, w), kernel_size=(self.k, self.k), padding=(1, 1))

        # Third Transformer Block
        xt3 = self.unFold(xt2_folded)
        xt3_norm = self.norm(xt3.transpose(-2, -1)).transpose(-2, -1)
        xt3_transformed = self.emha3(xt3_norm) + xt3
        xt3_folded = F.fold(xt3_transformed, output_size=(h, w), kernel_size=(self.k, self.k), padding=(1, 1))  
        # Output layer
        out = x+self.conv(xt3_folded)

        return out

class Encode_model(nn.Module):
    def __init__(self, in_channels, n_blocks=4, scale=4):
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
        body = [EnhancedBB(in_channels=n_feats, out_channels=n_feats) for _ in range(n_blocks)]
        body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1))
        self.body = nn.Sequential(*body)
        self.conv = MyConv2d(in_nc=in_channels, out_nc=n_feats, kernel_size=1)

    def forward(self, x):
        out1 = self.head(x)
        out = self.body(out1)
        return out1, out + out1


class Generator_model(nn.Module):
    def __init__(self, args):
        super(Generator_model, self).__init__()
        # self.scale = int(args.scale)
        self.scale = 4
        if self.scale == 2:
            res_scale = 1
            n_feats = 64
            self.upsample = nn.Sequential(nn.Conv2d(n_feats, n_feats*4, kernel_size=3, stride=1, padding=1),
                                      nn.PixelShuffle(2),
                                      nn.Conv2d(n_feats, 3, kernel_size=3, stride=1, padding=1))
        elif self.scale == 4:
            res_scale = 0.1
            n_feats = 128
            self.upsample = nn.Sequential(nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, stride=1, padding=1),
                                      nn.PixelShuffle(2),
                                      nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, stride=1, padding=1),
                                      nn.PixelShuffle(2),
                                      nn.Conv2d(n_feats, 3, kernel_size=3, stride=1, padding=1)
                                      )
        elif self.scale == 8:
            res_scale = 0.1
            n_feats = 128
            self.upsample = nn.Sequential(nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, stride=1, padding=1),
                                      nn.PixelShuffle(2),
                                      nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, stride=1, padding=1),
                                      nn.PixelShuffle(2),
                                      nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, stride=1, padding=1),
                                      nn.PixelShuffle(2),
                                      nn.Conv2d(n_feats, 3, kernel_size=3, stride=1, padding=1)
                                      )
        self.encode = Encode_model(in_channels=3, n_blocks=5, scale=self.scale)
        self.AF = QKVFusionBlock(n_feats = n_feats)

    def forward(self, x):
        lfe, hfe = self.encode(x)
        upsampled_img = self.upsample(hfe)
        out = self.AF(x, upsampled_img)
        return out

# class AFBlock(nn.Module):
#     def __init__(self, n_feats, channels=3):
#         """
#         Adaptive Fusion Block for super-resolution
#         Args:
#             channels (int): Number of input channels
#         """
#         super(AFBlock, self).__init__()
        
#         self.channels = channels
#         self.clp_channels = 2*(channels + n_feats)
        
#         # Conv-LReLU-AvgP operator
#         self.clp = nn.Sequential(
#             nn.Conv2d(self.clp_channels, channels, kernel_size=3, padding=1),
#             nn.ELU(inplace=True),
#             nn.AdaptiveMaxPool2d(1)  # Global average pooling
#         )
        
#         # Linear transformations for weight generation
#         self.Ph = nn.Linear(channels, channels)
#         self.Pl = nn.Linear(channels, channels)
        
#     def forward(self, lr_img, lfe, hfe, hr_img):
#         """
#         Forward pass of the AF Block
#         Args:
#             lr_img (torch.Tensor): Input LR image [B, C, H, W]
#             high_level_features (torch.Tensor): High-level features from previous layers [B, C, H, W]
#         Returns:
#             torch.Tensor: Super-resolved output
#         """
#         # Equation 7: Concatenate LR image and high-level features
#         lr_img = F.interpolate(lr_img,scale_factor=4,mode='bicubic')
#         lfe = F.interpolate(lfe,scale_factor=4,mode='bicubic')
#         hfe = F.interpolate(hfe,scale_factor=4,mode='bicubic')

#         # concatenat all the branches
#         concat_features = torch.cat([lr_img, lfe, hfe, hr_img], dim=1)
#         # print(f"lr image shape: {lr_img.shape}, lfe shape: {lfe.shape}, hfe shape: {hfe.shape}, hr_img shape: {hr_img.shape}")
        
#         # Equation 8: Generate initial weights through Conv-LReLU-AvgP
#         w_init = self.clp(concat_features)  # [B, C, 1, 1]
#         # print(w_init.shape,'shape of winit')
#         w_init = w_init.squeeze(-1).squeeze(-1)  # [B, C]
        
#         # Equations 9-10: Generate separate weights for LR and high-level features
#         w_h = self.Ph(w_init)  # [B, C]
#         w_l = self.Pl(w_init)  # [B, C]
        
#         # Apply softmax to get final weights
#         weights = torch.stack([w_h, w_l], dim=-1)  # [B, C, 2]
#         weights = F.softmax(weights, dim=-1)
#         w_h, w_l = weights[..., 0], weights[..., 1]  # [B, C]
        
#         # Equation 11: Generate final output by weighted combination
#         # Reshape weights for broadcasting
#         w_h = w_h.view(-1, self.channels, 1, 1)
#         w_l = w_l.view(-1, self.channels, 1, 1)
        
#         output = w_h * hr_img + w_l * lr_img
        
#         return output
class QKVFusionBlock(nn.Module):
    def __init__(self, n_feats, channels=3, heads=4):
        """
        Simplified QKV Fusion Block
        Args:
            n_feats (int): Number of feature channels
            channels (int): Number of input image channels
            heads (int): Number of attention heads
        """
        super(QKVFusionBlock, self).__init__()
        
        self.channels = channels
        self.heads = heads
        self.head_dim = n_feats // heads
        
        # Weight generation for Query and Key
        self.clp = nn.Sequential(
            nn.Conv2d(2*channels, channels, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.AdaptiveMaxPool2d(1)
        )
        
        # Projections for Query, Key, and Value
        self.query_proj = nn.Linear(channels, n_feats)
        self.key_proj = nn.Linear(channels, n_feats)
        self.value_proj = nn.Linear(channels, n_feats)
        
        # Output projection
        self.out_proj = nn.Linear(n_feats, channels)
        
        # Attention scaling
        self.scale = self.head_dim ** -0.5
        
        # Dropout for regularization
        self.attn_dropout = nn.Dropout(0.1)
    
    def forward(self, lr_img, hr_img):
        """
        Forward pass of the QKV Fusion Block
        Args:
            lr_img (torch.Tensor): Input LR image [B, C, H, W]
            hr_img (torch.Tensor): High-resolution/Super-resolved image [B, C, H, W]
        Returns:
            torch.Tensor: Attention-weighted output
        """
        # Interpolate low-res image to match high-res image
        lr_img = F.interpolate(lr_img, scale_factor=4, mode='bicubic')
        
        # Concatenate LR and HR images for weight generation
        concat_features = torch.cat([lr_img, hr_img], dim=1)
        
        # Generate initial weights through Conv-LReLU-AvgP
        w_init = self.clp(concat_features)  # [B, C, 1, 1]
        w_init = w_init.squeeze(-1).squeeze(-1)  # [B, C]
        
        # Generate Query and Key from w_init
        query = self.query_proj(w_init)  # [B, n_feats]
        key = self.key_proj(w_init)      # [B, n_feats]
        
        # Project Value from hr_img (same as sr_img)
        value_global = F.adaptive_avg_pool2d(hr_img, (1, 1)).squeeze(-1).squeeze(-1)
        value = self.value_proj(value_global)  # [B, n_feats]
        
        # Multi-head attention preparation
        query = query.view(-1, self.heads, self.head_dim).transpose(0, 1)
        key = key.view(-1, self.heads, self.head_dim).transpose(0, 1)
        value = value.view(-1, self.heads, self.head_dim).transpose(0, 1)
        
        # Compute attention scores
        attn_scores = (query @ key.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        
        # Compute weighted value
        weighted_value = attn_probs @ value
        weighted_value = weighted_value.transpose(0, 1).contiguous().view(-1, self.heads * self.head_dim)
        
        # Output projection
        output_features = self.out_proj(weighted_value)
        
        # Reshape output features for image fusion
        output_features = output_features.view(-1, self.channels, 1, 1)
        
        # Weighted fusion of images
        output = output_features * hr_img + (1 - output_features) * lr_img
        
        return output

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
    
# Patch Discriminator for gan
    
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

class Patch_Discriminator(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='lrelu', mode='CNA',out_feat=256):
        super(Patch_Discriminator, self).__init__()
        # 192, 64 (12,512)
        conv0 = MyConv2d(in_nc, base_nf, kernel_size=4, stride=2, norm_type=None, act_type=act_type, \
            mode=mode)
        conv1 = MyConv2d(base_nf, 2*base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 96, 64 (6,64)
        conv2 = MyConv2d(2*base_nf, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv3 = MyConv2d(base_nf*4, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 48, 128 (3,128)
        conv4 = MyConv2d(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv5 = MyConv2d(base_nf*8, 1, kernel_size=4, norm_type=None, \
            act_type='sigm', mode=mode)
        
        
        self.features = sequential(conv0, conv1, conv2, conv3, conv4, conv5)

        #self.gap = nn.AdaptiveAvgPool2d((1,1))
        #self.classifier = nn.Sequential(
        #    nn.Linear(base_nf*8, 512), nn.LeakyReLU(0.2, True), nn.Linear(512, out_feat))

    def forward(self, x):
        x = self.features(x)
        #x = x.view(x.size(0), -1)
        #x = self.classifier(x)
        return x