#TESTING PRETRAINED RES34 WITH TRANSFER LEARNING
import torch.nn as nn
from torchvision import models

   
#Using pretrained res34 and fine-tuned on the decoder    
class FCN(nn.Module):   
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        #pre-trained encoder
        resnet = models.resnet34(pretrained=True)
        self.features =  nn.Sequential(*(list(resnet.children())[:-2]))   #remove the fc layer and max pool layer
        self.relu    = nn.ReLU(inplace=True)
        #decode
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)

    def forward(self, x):
        
        # Encoder
        x5 =  self.features(x)
        # Decoder
        y1 = self.bn1(self.relu(self.deconv1(x5)))
        y2 = self.bn2(self.relu(self.deconv2(y1)))
        y3 = self.bn3(self.relu(self.deconv3(y2)))
        y4 = self.bn4(self.relu(self.deconv4(y3)))
        y5 = self.bn5(self.relu(self.deconv5(y4)))    
        score = self.classifier(y5)                   
        # for param in self.features.parameters():   freeze parameters
        #   param.requires_grad = False

        return score  # size=(N, n_class, x.H/1, x.W/1)


#This is different from the basic function weight initiation, we dont want to change the weight of pretrained parameters. 
def init_weights(m):
    if isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases   

