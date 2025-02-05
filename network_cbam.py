import torch
import torch.nn as nn
from timm.models.layers import DropPath
from CBAM import ChannelAttention, SpatialAttention
import torch.nn.functional as F

### MODEL
 
class AttentionGate(nn.Module):
    def __init__(self, in_dim, coarser_dim, hidden_dim) -> None:
        super(AttentionGate, self).__init__()
        
        self.GridGateSignal_generator = nn.Sequential(
                nn.Conv2d(coarser_dim, coarser_dim, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(coarser_dim),
                nn.ReLU()
            )
        
        # input feature x // the gating signal from a coarser scale
        # gating dim == in_dim*2
        self.w_x = nn.Conv2d(in_dim, hidden_dim, kernel_size=(2,2), stride=(2,2), padding=0, bias=False)
        self.w_g = nn.Conv2d(coarser_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(hidden_dim, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs, coarser): # coarser : query (이전 feature), inputs : encoder feature
        query = self.GridGateSignal_generator(coarser) # (B, 256, H/8, W/8)

        proj_x = self.w_x(inputs) # (B, 128, H/8, W/8)
        proj_g = self.w_g(query) # (B, 128, H/8, W/8)

        addtive = F.relu(proj_x + proj_g) # (B, 128, H/8, W/8)
        attn_coef = self.psi(addtive) # (B, 1, H/8, W/8)

        attn_coef = F.upsample(attn_coef, inputs.size()[2:], mode='bilinear') # (B, 1, H/4, W/4)

        return attn_coef

class Initial_block(nn.Module):
    def __init__(self, channels, group_num):
        super(Initial_block, self).__init__()

        self.gn1    = nn.GroupNorm(num_groups=group_num, num_channels=channels)
        self.act1   = nn.ReLU()
        self.conv1  = nn.Conv2d(channels, channels, 3, stride=1, padding=1)

        self.gn2    = nn.GroupNorm(num_groups=group_num, num_channels=channels)
        self.act2   = nn.ReLU()
        self.conv2  = nn.Conv2d(channels, channels, 3, stride=1, padding=1, groups=channels)

        self.gn3    = nn.GroupNorm(num_groups=group_num, num_channels=channels)
        self.act3   = nn.ReLU()
        self.conv3  = nn.Conv2d(channels, channels, 1, stride=1, padding=0)

        self.gn4    = nn.GroupNorm(num_groups=group_num, num_channels=channels)
        self.act4   = nn.ReLU()
        self.conv4  = nn.Conv2d(channels, channels, 3, stride=1, padding=1, groups=channels)

    def forward(self, x):
        y   = self.conv1(self.act1(self.gn1(x)))
        y   = self.conv2(self.act2(self.gn2(y)))
        y   = self.conv3(self.act3(self.gn3(y)))
        y   = self.conv4(self.act4(self.gn4(y)))
        return y


class LargeKernelReparam(nn.Module):
    def __init__(self, channels, kernel, small_kernels=()):
        super(LargeKernelReparam, self).__init__()

        self.dw_large = nn.Conv2d(channels, channels, kernel, padding=kernel//2, groups=channels)

        self.small_kernels = small_kernels
        for k in self.small_kernels:
            setattr(self, f"dw_small_{k}", nn.Conv2d(channels, channels, k, padding=k//2, groups=channels))

    def forward(self, in_p):
        out_p = self.dw_large(in_p)
        for k in self.small_kernels:
            out_p += getattr(self, f"dw_small_{k}")(in_p)
        return out_p        


class encblock(nn.Module):
    def __init__(self, channels, group_num, kernel=13, small_kernels=(5,), mlp_ratio=4.0, drop=0.3, drop_path=0.5):
        super(encblock, self).__init__()
        self.kernel         = kernel
        self.small_kernels  = small_kernels
        self.drop           = drop
        self.drop_path      = drop_path

        self.gn1            = nn.GroupNorm(num_groups=group_num, num_channels=channels)
        self.act1           = nn.ReLU()
        self.conv1          = nn.Conv2d(channels, channels, 1, stride=1, padding=0)

        self.gn2            = nn.GroupNorm(num_groups=group_num, num_channels=channels)
        self.act2           = nn.ReLU()
        self.lkr2           = LargeKernelReparam(channels, self.kernel, self.small_kernels)

        self.gn3            = nn.GroupNorm(num_groups=group_num, num_channels=channels)
        self.act3           = nn.ReLU()
        self.conv3          = nn.Conv2d(channels, channels, 1, stride=1, padding=0)

        self.gn4            = nn.GroupNorm(num_groups=group_num, num_channels=channels)
        self.act4           = nn.GELU()
        self.mlp4           = nn.Conv2d(channels, int(channels*mlp_ratio), 1, stride=1, padding=0)

        self.gn5            = nn.GroupNorm(num_groups=group_num, num_channels=int(channels*mlp_ratio))
        self.act5           = nn.GELU()
        self.mlp5           = nn.Conv2d(int(channels*mlp_ratio), channels, 1, stride=1, padding=0)

        self.dropout        = nn.Dropout(self.drop)
        self.droppath       = DropPath(self.drop_path) if self.drop_path > 0. else nn.Identity()

    def forward(self, x):
        y   = self.conv1(self.act1(self.gn1(x)))
        y   = self.lkr2(self.act2(self.gn2(y)))
        y   = self.conv3(self.act3(self.gn3(x)))
        x   = x + self.droppath(y)

        y   = self.mlp4(self.act4(self.gn4(x)))
        y   = self.dropout(y)
        y   = self.mlp5(self.act5(self.gn5(y)))
        y   = self.dropout(y)
        x   = x + self.droppath(y)
        return x


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, group_num):
        super(DownSample, self).__init__()

        self.gn1    = nn.GroupNorm(num_groups=group_num, num_channels=in_channels)
        self.act1   = nn.ReLU()
        self.conv1  = nn.Conv2d(in_channels, out_channels, 1)

        self.gn2    = nn.GroupNorm(num_groups=group_num, num_channels=out_channels)
        self.act2   = nn.ReLU()
        self.conv2  = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, groups=out_channels) #안되면 group 뺴고 해보기
    
    def forward(self, x):
        y   = self.conv1(self.act1(self.gn1(x)))
        y   = self.conv2(self.act2(self.gn2(y)))
        return y


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, group_num):
        super(UpSample, self).__init__()

        self.gn1    = nn.GroupNorm(num_groups=group_num, num_channels=in_channels)
        self.act1   = nn.ReLU()
        self.conv1  = nn.Conv2d(in_channels, out_channels, 1)

        self.gn2    = nn.GroupNorm(num_groups=group_num, num_channels=out_channels)
        self.act2   = nn.ReLU()
        self.up2    = nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2, groups=out_channels) #안되면 groups 빼고 해보기

    def forward(self, x):
        y   = self.conv1(self.act1(self.gn1(x)))
        y   = self.up2(self.act2(self.gn2(y)))
        return y


class decblock(nn.Module):
    def __init__(self, in_channels, out_channels, group_num):
        super(decblock, self).__init__()

        self.gn1    = nn.GroupNorm(num_groups=group_num, num_channels=in_channels)
        self.relu1  = nn.ReLU()
        self.conv1  = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)

        self.gn2    = nn.GroupNorm(num_groups=group_num, num_channels=out_channels)
        self.relu2  = nn.ReLU()
        self.conv2  = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=out_channels)

    def forward(self, x):
        y   = self.conv1(self.relu1(self.gn1(x)))
        y   = self.conv2(self.relu2(self.gn2(y)))
        return y

class RLKunet(nn.Module):
    def __init__(self, in_channels, out_channels=2, features=64, group_num=7):
        super(RLKunet, self).__init__()

        self.init_conv  = nn.Conv2d(in_channels, features, 1, stride=1, padding=0)
        self.init_block = Initial_block(features, group_num)

        self.encoder1_1 = encblock(features, group_num, drop_path=0.0)
        self.encoder1_2 = encblock(features, group_num, drop_path=0.0)
        self.down1      = DownSample(features, features*2, group_num)

        self.encoder2_1 = encblock(features*2, group_num, drop_path=0.2)
        self.encoder2_2 = encblock(features*2, group_num, drop_path=0.2)
        self.down2      = DownSample(features*2, features*4, group_num)

        self.encoder3_1 = encblock(features*4, group_num, drop_path=0.3)
        self.encoder3_2 = encblock(features*4, group_num, drop_path=0.3)
        self.down3      = DownSample(features*4, features*8, group_num)

        self.encoder4_1 = encblock(features*8, group_num, drop_path=0.5)
        self.encoder4_2 = encblock(features*8, group_num, drop_path=0.5)
        self.encoder4_3 = encblock(features*8, group_num, drop_path=0.5)
        self.encoder4_4 = encblock(features*8, group_num, drop_path=0.5)

        self.up3        = UpSample(features*8, features*4, group_num)
        self.decoder3_1 = decblock(features*(4+4), features*4, group_num)
        self.decoder3_2 = decblock(features*4, features*4, group_num)

        self.up2        = UpSample(features*4, features*2, group_num)
        self.decoder2_1 = decblock(features*(2+2), features*2, group_num)
        self.decoder2_2 = decblock(features*2, features*2, group_num)

        self.up1        = UpSample(features*2, features, group_num)
        self.decoder1_1 = decblock(features*(1+1), features, group_num)
        self.decoder1_2 = decblock(features, features, group_num)

        self.conv       = nn.Conv2d(features, out_channels, 1, stride=1 , padding=0)
        self.softmax    = nn.Softmax(dim=1)

        # highlight foreground
        self.hfconv4    = nn.Conv2d(features*8, out_channels, 1, stride=1, padding=0)
        self.hfconv3    = nn.Conv2d(features*4, out_channels, 1, stride=1, padding=0)
        self.hfconv2    = nn.Conv2d(features*2, out_channels, 1, stride=1, padding=0)
        
        # CBAM
        self.ca3        = ChannelAttention(in_planes=features*4)
        self.sa3        = SpatialAttention()
        self.attn3      = AttentionGate(in_dim=features*4, coarser_dim=features*8, hidden_dim=features*4)
        
        self.ca2        = ChannelAttention(in_planes=features*2)
        self.sa2        = SpatialAttention()
        self.attn2      = AttentionGate(in_dim=features*2, coarser_dim=features*4, hidden_dim=features*2)
        
        self.ca1        = ChannelAttention(in_planes=features)
        self.sa1        = SpatialAttention()
        self.attn1      = AttentionGate(in_dim=features, coarser_dim=features*2, hidden_dim=features*1)


    def forward(self, x):                       # batch, 1, H, W
        enc0_1      = self.init_conv(x)         # batch, 64, H, W
        enc0_2      = self.init_block(enc0_1)   # batch, 64, H, W

        enc1_1      = self.encoder1_1(enc0_2)   # batch, 64, H, W
        enc1_2      = self.encoder1_2(enc1_1)   # batch, 64, H, W
        dwn1        = self.down1(enc1_2)        # batch, 128, H/2, W/2

        enc2_1      = self.encoder2_1(dwn1)     # batch, 128, H/2, W/2
        enc2_2      = self.encoder2_2(enc2_1)   # batch, 128, H/2, W/2
        dwn2        = self.down2(enc2_2)        # batch, 256, H/4, W/4

        enc3_1      = self.encoder3_1(dwn2)     # batch, 256, H/4, W/4
        enc3_2      = self.encoder3_2(enc3_1)   # batch, 256, H/4, W/4
        dwn3        = self.down3(enc3_2)        # batch, 512, H/8, W/8

        enc4_1      = self.encoder4_1(dwn3)     # batch, 512, H/8, W/8
        enc4_2      = self.encoder4_2(enc4_1)   # batch, 512, H/8, W/8
        enc4_3      = self.encoder4_3(enc4_2)   # batch, 512, H/8, W/8
        enc4_4      = self.encoder4_4(enc4_3)   # batch, 512, H/8, W/8

        up3         = self.up3(enc4_4)                  # batch, 256, H/4, W/4
        enc3_2_cbam = self.ca3(enc3_2) * enc3_2
        enc3_2_cbam = self.sa3(enc3_2_cbam) * enc3_2_cbam # batch, 256, H/4, W/4
        attn3       = self.attn3(enc3_2_cbam, enc4_4) * enc3_2_cbam # batch, 256, H/4, W/4
        concat3     = torch.cat((attn3, up3), dim=1)    # batch, 256+256, H/4, W/4
        dec3_1      = self.decoder3_1(concat3)          # batch, 256, H/4, W/4
        dec3_2      = self.decoder3_2(dec3_1)           # batch, 256, H/4, W/4

        up2         = self.up2(dec3_2)                  # batch, 128, H/2, W/2
        enc2_2_cbam = self.ca2(enc2_2) * enc2_2         # batch, 128, H/2, W/2
        enc2_2_cbam = self.sa2(enc2_2_cbam) * enc2_2_cbam # batch, 128, H/2, W/2
        attn2       = self.attn2(enc2_2_cbam, dec3_2) * enc2_2_cbam # batch, 128, H/2, W/2
        concat2     = torch.cat((attn2, up2), dim=1)   # batch, 128+128, H/2, W/2
        dec2_1      = self.decoder2_1(concat2)          # batch, 128, H/2, W/2
        dec2_2      = self.decoder2_2(dec2_1)           # batch, 128, H/2, W/2

        up1         = self.up1(dec2_2)                  # batch, 64, H, W
        enc1_2_cbam = self.ca1(enc1_2) * enc1_2
        enc1_2_cbam = self.sa1(enc1_2_cbam) * enc1_2_cbam
        attn1       = self.attn1(enc1_2_cbam, dec2_2) * enc1_2_cbam
        concat1     = torch.cat((attn1, up1), dim=1)   # batch, 64+64, H, W
        dec1_1      = self.decoder1_1(concat1)          # batch, 64, H, W
        dec1_2      = self.decoder1_2(dec1_1)           # batch, 64, H, W

        dec1_out    = self.conv(dec1_2)                 # batch, 2, H, W
        out1        = self.softmax(dec1_out)

        # highlight foreground
        out4        = self.softmax(self.hfconv4(enc4_4))
        out3        = self.softmax(self.hfconv3(dec3_2))
        out2        = self.softmax(self.hfconv2(dec2_2))
        return out4, out3, out2, out1

class RLKunet2(nn.Module):
    def __init__(self, in_channels, out_channels=2, features=64, group_num=7):
        super(RLKunet2, self).__init__()

        self.init_conv  = nn.Conv2d(in_channels, features, 1, stride=1, padding=0)
        self.init_block = Initial_block(features, group_num)

        self.encoder1_1 = encblock(features, group_num, drop_path=0.0)
        self.encoder1_2 = encblock(features, group_num, drop_path=0.0)
        self.down1      = DownSample(features, features*2, group_num)

        self.encoder2_1 = encblock(features*2, group_num, drop_path=0.2)
        self.encoder2_2 = encblock(features*2, group_num, drop_path=0.2)
        self.down2      = DownSample(features*2, features*4, group_num)

        self.encoder3_1 = encblock(features*4, group_num, drop_path=0.3)
        self.encoder3_2 = encblock(features*4, group_num, drop_path=0.3)
        self.encoder3_3 = encblock(features*4, group_num, drop_path=0.3)
        self.encoder3_4 = encblock(features*4, group_num, drop_path=0.3)

        self.up3        = UpSample(features*8, features*4, group_num)

        self.up2        = UpSample(features*4, features*2, group_num)
        self.decoder2_1 = decblock(features*(2+2), features*2, group_num)
        self.decoder2_2 = decblock(features*2, features*2, group_num)

        self.up1        = UpSample(features*2, features, group_num)
        self.decoder1_1 = decblock(features*(1+1), features, group_num)
        self.decoder1_2 = decblock(features, features, group_num)

        self.conv       = nn.Conv2d(features, out_channels, 1, stride=1 , padding=0)
        self.softmax    = nn.Softmax(dim=1)

        # highlight foreground
        self.hfconv4    = nn.Conv2d(features*8, out_channels, 1, stride=1, padding=0)
        self.hfconv3    = nn.Conv2d(features*4, out_channels, 1, stride=1, padding=0)
        self.hfconv2    = nn.Conv2d(features*2, out_channels, 1, stride=1, padding=0)
        
        # CBAM
        self.ca3        = ChannelAttention(in_planes=features*4)
        self.sa3        = SpatialAttention()
        self.attn3      = AttentionGate(in_dim=features*4, coarser_dim=features*8, hidden_dim=features*4)
        
        self.ca2        = ChannelAttention(in_planes=features*2)
        self.sa2        = SpatialAttention()
        self.attn2      = AttentionGate(in_dim=features*2, coarser_dim=features*4, hidden_dim=features*2)
        
        self.ca1        = ChannelAttention(in_planes=features)
        self.sa1        = SpatialAttention()
        self.attn1      = AttentionGate(in_dim=features, coarser_dim=features*2, hidden_dim=features*1)


    def forward(self, x):                       # batch, 1, H, W
        enc0_1      = self.init_conv(x)         # batch, 64, H, W
        enc0_2      = self.init_block(enc0_1)   # batch, 64, H, W

        enc1_1      = self.encoder1_1(enc0_2)   # batch, 64, H, W
        enc1_2      = self.encoder1_2(enc1_1)   # batch, 64, H, W
        dwn1        = self.down1(enc1_2)        # batch, 128, H/2, W/2

        enc2_1      = self.encoder2_1(dwn1)     # batch, 128, H/2, W/2
        enc2_2      = self.encoder2_2(enc2_1)   # batch, 128, H/2, W/2
        dwn2        = self.down2(enc2_2)        # batch, 256, H/4, W/4

        enc3_1      = self.encoder3_1(dwn2)     # batch, 256, H/4, W/4
        enc3_2      = self.encoder3_2(enc3_1)   # batch, 256, H/4, W/4
        enc3_3      = self.encoder3_3(enc3_2)   # batch, 256, H/4, W/4
        enc3_4      = self.encoder3_4(enc3_3)   # batch, 256, H/4, W/4

        up2         = self.up2(enc3_4)                  # batch, 128, H/2, W/2
        enc2_2_cbam = self.ca2(enc2_2) * enc2_2         # batch, 128, H/2, W/2
        enc2_2_cbam = self.sa2(enc2_2_cbam) * enc2_2_cbam # batch, 128, H/2, W/2
        attn2       = self.attn2(enc2_2_cbam, enc3_4) * enc2_2_cbam # batch, 128, H/2, W/2
        concat2     = torch.cat((attn2, up2), dim=1)   # batch, 128+128, H/2, W/2
        dec2_1      = self.decoder2_1(concat2)          # batch, 128, H/2, W/2
        dec2_2      = self.decoder2_2(dec2_1)           # batch, 128, H/2, W/2

        up1         = self.up1(dec2_2)                  # batch, 64, H, W
        enc1_2_cbam = self.ca1(enc1_2) * enc1_2
        enc1_2_cbam = self.sa1(enc1_2_cbam) * enc1_2_cbam
        attn1       = self.attn1(enc1_2_cbam, dec2_2) * enc1_2_cbam
        concat1     = torch.cat((attn1, up1), dim=1)   # batch, 64+64, H, W
        dec1_1      = self.decoder1_1(concat1)          # batch, 64, H, W
        dec1_2      = self.decoder1_2(dec1_1)           # batch, 64, H, W

        dec1_out    = self.conv(dec1_2)                 # batch, 2, H, W
        out1        = self.softmax(dec1_out)

        # highlight foreground
        out3        = self.softmax(self.hfconv3(enc3_4))
        out2        = self.softmax(self.hfconv2(dec2_2))
        #print(out3.shape, out2.shape, out1.shape)
        return out3, out2, out1

class RLKunet3(nn.Module):
    def __init__(self, in_channels, out_channels=2, features=64, group_num=7):
        super(RLKunet3, self).__init__()

        self.init_conv  = nn.Conv2d(in_channels, features, 1, stride=1, padding=0)
        self.init_block = Initial_block(features, group_num)

        self.encoder1_1 = encblock(features, group_num, drop_path=0.0)
        self.encoder1_2 = encblock(features, group_num, drop_path=0.1)
        self.down1      = DownSample(features, features*2, group_num)

        self.encoder2_1 = encblock(features*2, group_num, drop_path=0.2)
        self.encoder2_2 = encblock(features*2, group_num, drop_path=0.2)
        self.encoder2_3 = encblock(features*2, group_num, drop_path=0.3)
        self.encoder2_4 = encblock(features*2, group_num, drop_path=0.3)


        self.up1        = UpSample(features*2, features, group_num)
        self.decoder1_1 = decblock(features*(1+1), features, group_num)
        self.decoder1_2 = decblock(features, features, group_num)

        self.conv       = nn.Conv2d(features, out_channels, 1, stride=1 , padding=0)
        self.softmax    = nn.Softmax(dim=1)

        # highlight foreground
        self.hfconv2    = nn.Conv2d(features*2, out_channels, 1, stride=1, padding=0)
        
        # CBAM
        
        self.ca2        = ChannelAttention(in_planes=features*2)
        self.sa2        = SpatialAttention()
        self.attn2      = AttentionGate(in_dim=features*2, coarser_dim=features*4, hidden_dim=features*2)
        
        self.ca1        = ChannelAttention(in_planes=features)
        self.sa1        = SpatialAttention()
        self.attn1      = AttentionGate(in_dim=features, coarser_dim=features*2, hidden_dim=features*1)


    def forward(self, x):                       # batch, 1, H, W
        enc0_1      = self.init_conv(x)         # batch, 64, H, W
        enc0_2      = self.init_block(enc0_1)   # batch, 64, H, W

        enc1_1      = self.encoder1_1(enc0_2)   # batch, 64, H, W
        enc1_2      = self.encoder1_2(enc1_1)   # batch, 64, H, W
        dwn1        = self.down1(enc1_2)        # batch, 128, H/2, W/2

        enc2_1      = self.encoder2_1(dwn1)     # batch, 128, H/2, W/2
        enc2_2      = self.encoder2_2(enc2_1)   # batch, 128, H/2, W/2
        enc2_3      = self.encoder2_3(enc2_2)   # batch, 128, H/2, W/2
        enc2_4      = self.encoder2_4(enc2_3)   # batch, 128, H/2, W/2

        up1         = self.up1(enc2_4)                  # batch, 64, H, W
        enc1_2_cbam = self.ca1(enc1_2) * enc1_2
        enc1_2_cbam = self.sa1(enc1_2_cbam) * enc1_2_cbam
        attn1       = self.attn1(enc1_2_cbam, enc2_4) * enc1_2_cbam
        concat1     = torch.cat((attn1, up1), dim=1)   # batch, 64+64, H, W
        dec1_1      = self.decoder1_1(concat1)          # batch, 64, H, W
        dec1_2      = self.decoder1_2(dec1_1)           # batch, 64, H, W

        dec1_out    = self.conv(dec1_2)                 # batch, 2, H, W
        out1        = self.softmax(dec1_out)

        # highlight foreground
        out2        = self.softmax(self.hfconv2(enc2_4))
        #print(out3.shape, out2.shape, out1.shape)
        return out2, out1

class RLKunet4(nn.Module):
    def __init__(self, in_channels, out_channels=2, features=64, group_num=7):
        super(RLKunet4, self).__init__()

        self.init_conv  = nn.Conv2d(in_channels, features, 1, stride=1, padding=0)
        self.init_block = Initial_block(features, group_num)

        self.encoder1_1 = encblock(features, group_num, drop_path=0.0)
        self.encoder1_2 = encblock(features, group_num, drop_path=0.1)
        self.encoder1_3 = encblock(features, group_num, drop_path=0.1)
        self.encoder1_4 = encblock(features, group_num, drop_path=0.1)
        self.down1      = DownSample(features, features*2, group_num)

        self.encoder2_1 = encblock(features*2, group_num, drop_path=0.2)
        self.encoder2_2 = encblock(features*2, group_num, drop_path=0.2)
        self.encoder2_3 = encblock(features*2, group_num, drop_path=0.3)
        self.encoder2_4 = encblock(features*2, group_num, drop_path=0.3)
        self.encoder2_5 = encblock(features*2, group_num, drop_path=0.4)
        self.encoder2_6 = encblock(features*2, group_num, drop_path=0.4)


        self.up1        = UpSample(features*2, features, group_num)
        self.decoder1_1 = decblock(features*(1+1), features, group_num)
        self.decoder1_2 = decblock(features, features, group_num)
        self.decoder1_3 = decblock(features, features, group_num)
        self.decoder1_4 = decblock(features, features, group_num)

        self.conv       = nn.Conv2d(features, out_channels, 1, stride=1 , padding=0)
        self.softmax    = nn.Softmax(dim=1)

        # highlight foreground
        self.hfconv2    = nn.Conv2d(features*2, out_channels, 1, stride=1, padding=0)
        
        # CBAM
        
        self.ca2        = ChannelAttention(in_planes=features*2)
        self.sa2        = SpatialAttention()
        self.attn2      = AttentionGate(in_dim=features*2, coarser_dim=features*4, hidden_dim=features*2)
        
        self.ca1        = ChannelAttention(in_planes=features)
        self.sa1        = SpatialAttention()
        self.attn1      = AttentionGate(in_dim=features, coarser_dim=features*2, hidden_dim=features*1)


    def forward(self, x):                       # batch, 1, H, W
        enc0_1      = self.init_conv(x)         # batch, 64, H, W
        enc0_2      = self.init_block(enc0_1)   # batch, 64, H, W

        enc1_1      = self.encoder1_1(enc0_2)   # batch, 64, H, W
        enc1_2      = self.encoder1_2(enc1_1)   # batch, 64, H, W
        dwn1        = self.down1(enc1_2)        # batch, 128, H/2, W/2

        enc2_1      = self.encoder2_1(dwn1)     # batch, 128, H/2, W/2
        enc2_2      = self.encoder2_2(enc2_1)   # batch, 128, H/2, W/2
        enc2_3      = self.encoder2_3(enc2_2)   # batch, 128, H/2, W/2
        enc2_4      = self.encoder2_4(enc2_3)   # batch, 128, H/2, W/2
        enc2_5      = self.encoder2_5(enc2_4)
        enc2_6      = self.encoder2_6(enc2_5)

        up1         = self.up1(enc2_6)                  # batch, 64, H, W
        enc1_2_cbam = self.ca1(enc1_2) * enc1_2
        enc1_2_cbam = self.sa1(enc1_2_cbam) * enc1_2_cbam
        attn1       = self.attn1(enc1_2_cbam, enc2_6) * enc1_2_cbam
        concat1     = torch.cat((attn1, up1), dim=1)   # batch, 64+64, H, W
        dec1_1      = self.decoder1_1(concat1)          # batch, 64, H, W
        dec1_2      = self.decoder1_2(dec1_1)           # batch, 64, H, W
        dec1_3      = self.decoder1_3(dec1_2)
        dec1_4      = self.decoder1_4(dec1_3)

        dec1_out    = self.conv(dec1_4)                 # batch, 2, H, W
        out1        = self.softmax(dec1_out)

        # highlight foreground
        out2        = self.softmax(self.hfconv2(enc2_6))
        #print(out3.shape, out2.shape, out1.shape)
        return out2, out1

class RLKunet5(nn.Module):
    def __init__(self, in_channels, out_channels=2, features=64, group_num=7):
        super(RLKunet5, self).__init__()

        self.init_conv  = nn.Conv2d(in_channels, features, 1, stride=1, padding=0)
        self.init_block = Initial_block(features, group_num)

        self.encoder1_1 = encblock(features, group_num, drop_path=0.0)
        self.encoder1_2 = encblock(features, group_num, drop_path=0.1)
        self.encoder1_3 = encblock(features, group_num, drop_path=0.1)
        self.encoder1_4 = encblock(features, group_num, drop_path=0.1)
        self.down1      = DownSample(features, features*2, group_num)

        self.encoder2_1 = encblock(features*2, group_num, drop_path=0.2)
        self.encoder2_2 = encblock(features*2, group_num, drop_path=0.2)
        self.encoder2_3 = encblock(features*2, group_num, drop_path=0.3)
        self.encoder2_4 = encblock(features*2, group_num, drop_path=0.3)
        self.encoder2_5 = encblock(features*2, group_num, drop_path=0.4)
        self.encoder2_6 = encblock(features*2, group_num, drop_path=0.4)

        self.center_block = nn.Sequential(
            nn.Conv2d(features*2, features*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(features*4),
            nn.ReLU(inplace=True),

            nn.Conv2d(features*4, features*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features*2),
            nn.ReLU(inplace=True)
        )

        self.up1        = UpSample(features*2, features, group_num)
        self.decoder1_1 = decblock(features*(1+1), features, group_num)
        self.decoder1_2 = decblock(features, features, group_num)
        self.decoder1_3 = decblock(features, features, group_num)
        self.decoder1_4 = decblock(features, features, group_num)

        self.conv       = nn.Conv2d(features, out_channels, 1, stride=1 , padding=0)
        self.softmax    = nn.Softmax(dim=1)

        # highlight foreground
        self.hfconv2    = nn.Conv2d(features*2, out_channels, 1, stride=1, padding=0)
        
        # CBAM
        
        self.ca2        = ChannelAttention(in_planes=features*2)
        self.sa2        = SpatialAttention()
        self.attn2      = AttentionGate(in_dim=features*2, coarser_dim=features*4, hidden_dim=features*2)
        
        self.ca1        = ChannelAttention(in_planes=features)
        self.sa1        = SpatialAttention()
        self.attn1      = AttentionGate(in_dim=features, coarser_dim=features*2, hidden_dim=features*1)


    def forward(self, x):                       # batch, 1, H, W
        enc0_1      = self.init_conv(x)         # batch, 64, H, W
        enc0_2      = self.init_block(enc0_1)   # batch, 64, H, W

        enc1_1      = self.encoder1_1(enc0_2)   # batch, 64, H, W
        enc1_2      = self.encoder1_2(enc1_1)   # batch, 64, H, W
        dwn1        = self.down1(enc1_2)        # batch, 128, H/2, W/2

        enc2_1      = self.encoder2_1(dwn1)     # batch, 128, H/2, W/2
        enc2_2      = self.encoder2_2(enc2_1)   # batch, 128, H/2, W/2
        enc2_3      = self.encoder2_3(enc2_2)   # batch, 128, H/2, W/2
        enc2_4      = self.encoder2_4(enc2_3)   # batch, 128, H/2, W/2
        enc2_5      = self.encoder2_5(enc2_4)
        enc2_6      = self.encoder2_6(enc2_5)

        center_out  = self.center_block(enc2_6)  # 128 -> 256 -> 128


        up1         = self.up1(center_out)                  # batch, 64, H, W
        enc1_2_cbam = self.ca1(enc1_2) * enc1_2
        enc1_2_cbam = self.sa1(enc1_2_cbam) * enc1_2_cbam
        attn1       = self.attn1(enc1_2_cbam, enc2_6) * enc1_2_cbam
        concat1     = torch.cat((attn1, up1), dim=1)   # batch, 64+64, H, W
        dec1_1      = self.decoder1_1(concat1)          # batch, 64, H, W
        dec1_2      = self.decoder1_2(dec1_1)           # batch, 64, H, W
        dec1_3      = self.decoder1_3(dec1_2)
        dec1_4      = self.decoder1_4(dec1_3)

        dec1_out    = self.conv(dec1_4)                 # batch, 2, H, W
        out1        = self.softmax(dec1_out)

        # highlight foreground
        out2        = self.softmax(self.hfconv2(enc2_6))
        #print(out3.shape, out2.shape, out1.shape)
        return out2, out1

def initialize_weight(m):
    
    if type(m) == nn.Conv2d:
        
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    
    elif type(m) == nn.ConvTranspose2d:
        
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# device = torch.device('cuda:3')
# model = RLKunet(in_channels=3, out_channels=2, features=64, group_num=8).to(device)
# img = torch.randn(8, 3, 128, 128).to(device)

# y_pred4, y_pred3, y_pred2, y_pred1 = model(img)
# a = 1