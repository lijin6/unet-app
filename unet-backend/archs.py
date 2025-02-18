import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
__all__ = ['UNet', 'NestedUNet', 'TransUNet']

try:
    from flash_attn.flash_attention import FlashMHA
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3,deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output




class TransformerBlock(nn.Module):
    def __init__(self, in_channels, num_heads=2, ff_channels=None, use_flash_attn=True):
        super().__init__()
        self.embed_dim = in_channels
        ff_channels = ff_channels or in_channels

        # 修改1：使用正确的注意力头数和通道数
        if use_flash_attn and FLASH_ATTN_AVAILABLE:
            self.attn = FlashMHA(
                embed_dim=self.embed_dim,
                num_heads=num_heads,
                bias=False
            )
        else:
            self.attn = nn.MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=num_heads,
                batch_first=True
            )

        # 修改2：调整LayerNorm的维度处理
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        
        # 修改3：使用ConvFFN代替LinearFFN
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels, ff_channels, 1),
            nn.GELU(),
            nn.Conv2d(ff_channels, in_channels, 1)
        )

    def _forward_impl(self, x):
        batch_size, channels, h, w = x.size()
        
        # 修改4：正确的维度重排
        x_flat = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x_flat = x_flat.reshape(batch_size, h*w, channels)
        
        # 注意力计算
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        attn_out = attn_out.reshape(batch_size, h, w, channels)
        attn_out = attn_out.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # 修改5：先应用LayerNorm再reshape
        x = x + attn_out
        x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # FFN处理
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        return x

    def forward(self, x):
        # 修改6：显式传递use_reentrant参数
        return checkpoint.checkpoint(self._forward_impl, x, use_reentrant=False)
        
class TransUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False):
        super(TransUNet, self).__init__()
        
        nb_filter = [32, 64, 128, 256, 512]
        self.deep_supervision = deep_supervision

        # Encoder
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        # Transformer at the deepest layer
        self.transformer = TransformerBlock(nb_filter[4], num_heads=8, ff_channels=1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(nb_filter[4], nb_filter[4], kernel_size=2, stride=2)
        self.conv3_1 = VGGBlock(nb_filter[4]+nb_filter[3], nb_filter[3], nb_filter[3])
        self.up3 = nn.ConvTranspose2d(nb_filter[3], nb_filter[3], kernel_size=2, stride=2)
        self.conv2_1 = VGGBlock(nb_filter[3]+nb_filter[2], nb_filter[2], nb_filter[2])
        self.up2 = nn.ConvTranspose2d(nb_filter[2], nb_filter[2], kernel_size=2, stride=2)
        self.conv1_1 = VGGBlock(nb_filter[2]+nb_filter[1], nb_filter[1], nb_filter[1])
        self.up1 = nn.ConvTranspose2d(nb_filter[1], nb_filter[1], kernel_size=2, stride=2)
        self.conv0_1 = VGGBlock(nb_filter[1]+nb_filter[0], nb_filter[0], nb_filter[0])

        # Deep supervision
        if deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[3], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[2], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[1], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.conv0_0(x)
        x1 = self.conv1_0(F.max_pool2d(x0, 2))
        x2 = self.conv2_0(F.max_pool2d(x1, 2))
        x3 = self.conv3_0(F.max_pool2d(x2, 2))
        x4 = self.conv4_0(F.max_pool2d(x3, 2))

        # Apply Transformer at the deepest layer
        x4 = self.transformer(x4)

        # Decoder with skip connections
        d3 = self.conv3_1(torch.cat([x3, self.up4(x4)], 1))
        d2 = self.conv2_1(torch.cat([x2, self.up3(d3)], 1))
        d1 = self.conv1_1(torch.cat([x1, self.up2(d2)], 1))
        d0 = self.conv0_1(torch.cat([x0, self.up1(d1)], 1))

        if self.deep_supervision:
            out1 = F.interpolate(self.final1(d3), scale_factor=8, mode='bilinear')
            out2 = F.interpolate(self.final2(d2), scale_factor=4, mode='bilinear')
            out3 = F.interpolate(self.final3(d1), scale_factor=2, mode='bilinear')
            out4 = self.final4(d0)
            return [out1, out2, out3, out4]
        
        return self.final(d0)



