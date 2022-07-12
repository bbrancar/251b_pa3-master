import torch.nn as nn
from torchvision import models
import torch

   
class Res_Unet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        
        
        #encoder
        #conv 1
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            )
        self.conv_skip_1 = nn.Sequential(
             nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1),
            )
        
        
        #conv 2
        self.conv_block_2 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            )
        self.conv_skip_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm2d(64),
            )
        
        
        #conv 3
        self.conv_block_3 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            )
        self.conv_skip_3 = nn.Sequential(
             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1),
             nn.BatchNorm2d(128),
            )  
        
        
        #conv 4
        self.conv_block_4 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            )
        self.conv_skip_4 = nn.Sequential(
             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation=1),
             nn.BatchNorm2d(256),
            )   
        
        
        #conv 5
        self.conv_block_5 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            )
        self.conv_skip_5 = nn.Sequential(
             nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1),
             nn.BatchNorm2d(512),
            )   
        
         #conv 6
        self.conv_block_6 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            )
        self.conv_skip_6 = nn.Sequential(
              nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, dilation=1),
              nn.BatchNorm2d(1024),
            )   
        
        
        #decoder
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        #deconv0
        self.deconv0 = nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv0_res = nn.Sequential(
            nn.BatchNorm2d(1024+512),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024+512, 512, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            )
        self.deconv_skip_0 = nn.Sequential(
             nn.Conv2d(1024+512, 512, kernel_size=3, stride=1, padding=1, dilation=1),
             nn.BatchNorm2d(512),
            )  
        
        
        #deconv1
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv1_res = nn.Sequential(
            nn.BatchNorm2d(512+256),
            nn.ReLU(inplace=True),
            nn.Conv2d(512+256, 256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            )
        self.deconv_skip_1 = nn.Sequential(
             nn.Conv2d(512+256, 256, kernel_size=3, stride=1, padding=1, dilation=1),
             nn.BatchNorm2d(256),
            )  
        
        #deconv2
        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv2_res = nn.Sequential(
            nn.BatchNorm2d(256 + 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            )
        self.deconv_skip_2 = nn.Sequential(
             nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1, dilation=1),
             nn.BatchNorm2d(128),
            )  
        #deconv3
        self.deconv3 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv3_res = nn.Sequential(
            nn.BatchNorm2d(128 + 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            )
        self.deconv_skip_3 = nn.Sequential(
             nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1, dilation=1),
             nn.BatchNorm2d(64),
            ) 
        
        #deconv4
        self.deconv4 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv4_res = nn.Sequential(
            nn.BatchNorm2d(64 + 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            )
        self.deconv_skip_4 = nn.Sequential(
             nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1, dilation=1),
             nn.BatchNorm2d(32),
            ) 
        
        self.deconv5 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)

    def forward(self, x):
        #32,3,384,768
        x1 = self.conv_block_1(x) + self.conv_skip_1(x)

        #32, 32, 192, 384
        x2 = self.conv_block_2(x1) + self.conv_skip_2(x1)

        #32, 64, 96, 192
        x3 = self.conv_block_3(x2) + self.conv_skip_3(x2)

        #32, 128, 48, 96
        x4 = self.conv_block_4(x3) + self.conv_skip_4(x3)

        #32, 256, 24, 48
        x5 = self.conv_block_5(x4) + self.conv_skip_5(x4)

        #32, 512, 12, 24
        x6 = self.conv_block_6(x5) + self.conv_skip_6(x5)

        #32, 1024, 6, 12
        x6 = self.deconv0(x6)
      
        #32, 1024, 12, 24
        x7 = torch.cat([x6, x5], dim = 1)
      
        #32, 512, 12, 24
        x8 = self.deconv0_res(x7) + self.deconv_skip_0(x7)
      
        #32, 512, 12, 24
        x8 = self.deconv1(x8)
      
        #32, 512, 24, 48
        x9 = torch.cat([x8,x4], dim = 1)
      
        #32, 768, 24, 48
        x10 = self.deconv1_res(x9) + self.deconv_skip_1(x9)
      
        #32, 256, 24, 48
        x10 = self.deconv2(x10)  
      
        #32, 256, 48, 96
        x11 = torch.cat([x10,x3], dim = 1)
      
        #32, 384, 48, 96
        x12 = self.deconv2_res(x11) + self.deconv_skip_2(x11)
      
        #32, 128, 48, 96
        x12 = self.deconv3(x12)
      
        #32, 128, 96, 192
        x13 = torch.cat([x12,x2], dim = 1)
      
        #32, 192, 96, 192
        x14 = self.deconv3_res(x13) + self.deconv_skip_3(x13)
      
        #32, 64, 96, 192
        x14 = self.deconv4(x14)
      
        #32, 64, 192, 384
        x15 = torch.cat([x14,x1], dim = 1)
      
        #32, 96, 192, 384
        x16 = self.deconv4_res(x15) + self.deconv_skip_4(x15)
      
        #32, 32, 192, 384
        x16 = self.deconv5(x16)

        #32, 32, 384, 768
        score = self.classifier(x16)                   

        return score  # size=(N, n_class, x.H/1, x.W/1)
    
if __name__ == '__main__':
    x = torch.rand((32,3,384,768))
    model = Res_Unet(n_class = 10)
    model(x)
