#Implementation of U-Net model
import torch.nn as nn
   
class UNET(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class

        #Used after each bloc
        self.mpool_en = nn.MaxPool2d(2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=.2)

        #First enc patch
        self.en_conv1_1   = nn.Conv2d(3, 64, kernel_size=3, padding= "same")
        self.batch_1_1 = nn.BatchNorm2d(64)
        self.en_conv1_2   = nn.Conv2d(64, 64, kernel_size=3, padding= "same")
        self.batch_1_2 = nn.BatchNorm2d(64)
        
        #Second enc patch
        self.en_conv2_1   = nn.Conv2d(64, 128, kernel_size=3, padding= "same")
        self.batch_2_1 = nn.BatchNorm2d(128)
        self.en_conv2_2   = nn.Conv2d(128, 128, kernel_size=3, padding= "same")
        self.batch_2_2 = nn.BatchNorm2d(128)

        #Third enc patch
        self.en_conv3_1   = nn.Conv2d(128, 256, kernel_size=3, padding= "same")
        self.batch_3_1 = nn.BatchNorm2d(256)
        self.en_conv3_2   = nn.Conv2d(256, 256, kernel_size=3, padding= "same")
        self.batch_3_2 = nn.BatchNorm2d(256)

        #Fourth enc patch
        self.en_conv4_1   = nn.Conv2d(256, 512, kernel_size=3, padding= "same")
        self.batch_4_1 = nn.BatchNorm2d(512)
        self.en_conv4_2   = nn.Conv2d(512, 512, kernel_size=3, padding= "same")
        self.batch_4_2 = nn.BatchNorm2d(512)

        #Fifth enc patch
        self.en_conv5_1   = nn.Conv2d(512, 1024, kernel_size=3, padding= "same")
        self.batch_5_1 = nn.BatchNorm2d(1024)
        self.en_conv5_2   = nn.Conv2d(1024, 1024, kernel_size=3, padding= "same")
        self.batch_5_2 = nn.BatchNorm2d(1024)

        #Beginning of upward

        #First dec patch
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.de_conv1_1   = nn.Conv2d(1024, 512, kernel_size=3, padding= "same")
        self.batch_6_1 = nn.BatchNorm2d(512)
        self.de_conv1_2   = nn.Conv2d(512, 512, kernel_size=3, padding= "same")
        self.batch_6_2 = nn.BatchNorm2d(512)

        #Second dec patch
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.de_conv2_1   = nn.Conv2d(512, 256, kernel_size=3, padding= "same")
        self.batch_7_1 = nn.BatchNorm2d(256)
        self.de_conv2_2   = nn.Conv2d(256, 256, kernel_size=3, padding= "same")
        self.batch_7_2 = nn.BatchNorm2d(256)

        #Third dec patch
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.de_conv3_1   = nn.Conv2d(256, 128, kernel_size=3, padding= "same")
        self.batch_8_1 = nn.BatchNorm2d(128)
        self.de_conv3_2   = nn.Conv2d(128, 128, kernel_size=3, padding= "same")
        self.batch_8_2 = nn.BatchNorm2d(128)

        #Fourth dec patch
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.de_conv4_1   = nn.Conv2d(128, 64, kernel_size=3, padding= "same")
        self.batch_9_1 = nn.BatchNorm2d(64)
        self.de_conv4_2   = nn.Conv2d(64, 64, kernel_size=3, padding= "same")
        self.batch_9_2 = nn.BatchNorm2d(64)

        #Final out
        self.classifier = nn.Conv2d(64, self.n_class, kernel_size=1, padding= "same")

    def forward(self, x):
        
        # Encoder

        #Bloc1
        en_bloc1_out1 = self.drop(self.relu(self.batch_1_1(self.en_conv1_1(x))))
        en_bloc1_out2 = self.mpool_en(self.drop(self.relu(self.batch_1_2(self.en_conv1_2(en_bloc1_out1)))))
        
        #Bloc2
        en_bloc2_out1 = self.drop(self.relu(self.batch_2_1(self.en_conv2_1(en_bloc1_out2))))
        en_bloc2_out2 = self.mpool_en(self.drop(self.relu(self.batch_2_2(self.en_conv2_2(en_bloc2_out1)))))

        #Bloc3
        en_bloc3_out1 = self.drop(self.relu(self.batch_3_1(self.en_conv3_1(en_bloc2_out2))))
        en_bloc3_out2 = self.mpool_en(self.drop(self.relu(self.batch_3_2(self.en_conv3_2(en_bloc3_out1)))))

        #Bloc4
        en_bloc4_out1 = self.drop(self.relu(self.batch_4_1(self.en_conv4_1(en_bloc3_out2))))
        en_bloc4_out2 = self.mpool_en(self.drop(self.relu(self.batch_4_2(self.en_conv4_2(en_bloc4_out1)))))

        #Bloc5
        en_bloc5_out1 = self.drop(self.relu(self.batch_5_1(self.en_conv5_1(en_bloc4_out2))))
        en_bloc5_out2 = self.drop(self.relu(self.batch_5_2(self.en_conv5_2(en_bloc5_out1))))

        #Decoder

        #Bloc6
        de_bloc1_out1 = self.deconv1(en_bloc5_out2)
        concat_1 = torch.cat([en_bloc4_out2, de_bloc1_out1], dim=1)
        de_bloc1_out2 = self.drop(self.relu(self.batch_6_1(self.de_conv1_1(concat_1))))
        de_bloc1_out3 = self.drop(self.relu(self.batch_6_2(self.de_conv1_2(de_bloc1_out2))))

        #Bloc7
        de_bloc2_out1 = self.deconv2(de_bloc1_out3)
        concat_2 = torch.cat([en_bloc3_out2, de_bloc2_out1], dim=1)
        de_bloc2_out2 = self.drop(self.relu(self.batch_7_1(self.de_conv2_1(concat_2))))
        de_bloc2_out3 = self.drop(self.relu(self.batch_7_2(self.de_conv2_2(de_bloc2_out2))))

        #Bloc8
        de_bloc3_out1 = self.deconv3(de_bloc2_out3)
        concat_3 = torch.cat([en_bloc2_out2, de_bloc3_out1], dim=1)
        de_bloc3_out2 = self.drop(self.relu(self.batch_8_1(self.de_conv3_1(concat_3))))
        de_bloc3_out3 = self.drop(self.relu(self.batch_8_2(self.de_conv3_2(de_bloc3_out2))))

        #Bloc9
        de_bloc4_out1 = self.deconv4(de_bloc3_out3)
        concat_4 = torch.cat([en_bloc1_out2, de_bloc4_out1], dim=1)
        de_bloc4_out2 = self.drop(self.relu(self.batch_9_1(self.de_conv4_1(concat_4))))
        de_bloc4_out3 = self.drop(self.relu(self.batch_9_2(self.de_conv4_2(de_bloc4_out2))))

        #Final output
        score = self.classifier(de_bloc4_out3)
            

        return score  

