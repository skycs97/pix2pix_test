import torch
import torch.nn as nn
from torch.nn.modules import padding
from Pix2PixLayer import *

#Pix2Pix Generator모델
class Pix2PixGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, nfeature=64, dropout=0.5):
        super(Pix2PixGenerator, self).__init__()
        #make EncoderBlocks
        self.e1 = nn.Conv2d(in_channels=3, out_channels=nfeature, kernel_size=4, padding=1, stride=2)
        self.e2 = makeEncoderBlock(in_channels=nfeature, out_channels=nfeature*2)
        self.e3 = makeEncoderBlock(in_channels=nfeature*2, out_channels=nfeature*4)
        self.e4 = makeEncoderBlock(in_channels=nfeature*4, out_channels=nfeature*8)
        self.e5 = makeEncoderBlock(in_channels=nfeature*8, out_channels=nfeature*8)
        self.e6 = makeEncoderBlock(in_channels=nfeature*8, out_channels=nfeature*8)
        self.e7 = makeEncoderBlock(in_channels=nfeature*8, out_channels=nfeature*8)
        self.e8 = makeEncoderBlock(in_channels=nfeature*8, out_channels=nfeature*8, batch_norm=False)

        #make DecoderBlocks
        self.d1 = makeDecoderBlock(in_channels=nfeature*8, out_channels=nfeature*8, dropout=0.5)
        self.d2 = makeDecoderBlock(in_channels=nfeature*8*2, out_channels=nfeature*8, dropout=0.5)
        self.d3 = makeDecoderBlock(in_channels=nfeature*8*2, out_channels=nfeature*8, dropout=0.5)
        self.d4 = makeDecoderBlock(in_channels=nfeature*8*2, out_channels=nfeature*8, dropout=0)
        self.d5 = makeDecoderBlock(in_channels=nfeature*8*2, out_channels=nfeature*4, dropout=0)
        self.d6 = makeDecoderBlock(in_channels=nfeature*4*2, out_channels=nfeature*2, dropout=0)
        self.d7 = makeDecoderBlock(in_channels=nfeature*2*2, out_channels=nfeature, dropout=0)
        self.d8 = makeDecoderBlock(in_channels=nfeature*2, out_channels=out_channels, dropout=0, batch_norm=False)


    def forward(self, input_image):
        #forward Encoder
        e1_out = self.e1(input_image)
        e2_out = self.e2(e1_out)
        e3_out = self.e3(e2_out)
        e4_out = self.e4(e3_out)
        e5_out = self.e5(e4_out)
        e6_out = self.e6(e5_out)
        e7_out = self.e7(e6_out)
        e8_out = self.e8(e7_out)

        #forward Decoder
        d1_out = self.d1(e8_out)
        d1_out = torch.cat([d1_out, e7_out], dim=1)

        d2_out = self.d2(d1_out)
        d2_out = torch.cat([d2_out, e6_out], dim=1)

        d3_out = self.d3(d2_out)
        d3_out = torch.cat([d3_out, e5_out], dim=1)

        d4_out = self.d4(d3_out)
        d4_out = torch.cat([d4_out, e4_out], dim=1)

        d5_out = self.d5(d4_out)
        d5_out = torch.cat([d5_out, e3_out], dim=1)

        d6_out = self.d6(d5_out)
        d6_out = torch.cat([d6_out, e2_out], dim=1)

        d7_out = self.d7(d6_out)
        d7_out = torch.cat([d7_out, e1_out], dim=1)

        d8_out = self.d8(d7_out)

        #마지막 output은 tanh로 activation
        output_image = torch.tanh(d8_out)

        return output_image


#PatchGan 모델 생성
class Pix2PixDiscriminator(nn.Module):
    def __init__(self, in_channels=6, nfeature=64):
        super(Pix2PixDiscriminator, self).__init__()
        layers = list()
        layers.append(makeDiscriminatorBlock(in_channels=6, out_channels=nfeature, batch_norm=False))
        layers.append(makeDiscriminatorBlock(in_channels=nfeature, out_channels=nfeature*2))
        layers.append(makeDiscriminatorBlock(in_channels=nfeature*2, out_channels=nfeature*4))
        layers.append(makeDiscriminatorBlock(in_channels=nfeature*4, out_channels=nfeature*8, stride=1, padding=1))
        layers.append(nn.Conv2d(in_channels=nfeature*8, out_channels=1, kernel_size=4, stride=1, padding=1))

        self.discriminator = nn.Sequential(*layers)
        
    def forward(self, x):
        return torch.sigmoid(self.discriminator(x))