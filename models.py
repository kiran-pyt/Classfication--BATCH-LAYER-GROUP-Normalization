import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision import datasets,tranforms 
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from torchsummary import summary
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR



dropout_value = 0.1
class Net(nn.Module):
    def __init__(self, f, num=1):

        super(Net, self).__init__()
        self.f=f
        if self.f=='BN':

            self.convblock1=nn.Sequential(nn.Conv2d(in_channels=1,out_channels=10,kernel_size=(3,3),padding=0,bias=0),
                             nn.ReLU(),
                             nn.BatchNorm2d(10),
                             nn.Dropout(dropout_value)

                             ) # output_size = 26
            self.convblock2=nn.Sequential(nn.Conv2d(in_channels=10,out_channels=20,kernel_size=(3,3),padding=0,bias=0),
                             nn.ReLU(),
                             nn.BatchNorm2d(20),
                             nn.Dropout(dropout_value)
                             ) # output_size = 24
            self.convblock3=nn.Sequential(nn.Conv2d(in_channels=20,out_channels=10,kernel_size=(3,3),padding=0,bias=0),
                             nn.ReLU(),
                             nn.BatchNorm2d(10),
                             nn.Dropout(dropout_value)
                             )  # output_size = 22


            self.pool1=nn.MaxPool2d(2,2) # output_size = 11

            self.convblock4=nn.Sequential(nn.Conv2d(in_channels=10,out_channels=20,kernel_size=(1,1),padding=0,bias=0),
                             nn.ReLU(),
                             nn.BatchNorm2d(20),
                             nn.Dropout(dropout_value)
                             )   # output_size = 11

            self.convblock5=nn.Sequential(nn.Conv2d(in_channels=20,out_channels=10,kernel_size=(3,3),padding=0,bias=0),
                             nn.ReLU(),
                             nn.BatchNorm2d(10),
                             nn.Dropout(dropout_value)
                             )  # output_size = 9


            self.convblock6=nn.Sequential(nn.Conv2d(in_channels=10,out_channels=20,kernel_size=(3,3),padding=0,bias=0),
                             nn.ReLU(),
                             nn.BatchNorm2d(20),
                             nn.Dropout(dropout_value)
                             ) # output_size = 7
            self.convblock7=nn.Sequential(nn.Conv2d(in_channels=20,out_channels=10,kernel_size=(1,1),padding=0,bias=0),
                             nn.ReLU(),
                             nn.BatchNorm2d(10),
                             nn.Dropout(dropout_value)
                             ) # output_size =7

            self.gap = nn.Sequential(
           nn.AvgPool2d(kernel_size=7) # 7>> 9... nn.AdaptiveAvgPool((1, 1))
        ) # output_size = 1
            self.convblock8=nn.Sequential(nn.Conv2d(in_channels=10,out_channels=10,kernel_size=(1,1),padding=0,bias=0),
                             #nn.ReLU(),
                             #nn.BatchNorm2d(10),
                             #nn.Dropout(dropout_value)
                             ) # output_size =7
        elif self.f=='GN':
            self.num=num
            self.convblock1=nn.Sequential(nn.Conv2d(in_channels=1,out_channels=10,kernel_size=(3,3),padding=0,bias=0),
                             nn.ReLU(),
                             nn.GroupNorm(self.num,10),
                             nn.Dropout(dropout_value)

                             ) # output_size = 26
            self.convblock2=nn.Sequential(nn.Conv2d(in_channels=10,out_channels=20,kernel_size=(3,3),padding=0,bias=0),
                             nn.ReLU(),
                             nn.GroupNorm(self.num,20),
                             nn.Dropout(dropout_value)
                             ) # output_size = 24
            self.convblock3=nn.Sequential(nn.Conv2d(in_channels=20,out_channels=10,kernel_size=(3,3),padding=0,bias=0),
                            nn.ReLU(),
                            nn.GroupNorm(self.num,10),
                            nn.Dropout(dropout_value)
                             )  # output_size = 22


            self.pool1=nn.MaxPool2d(2,2) # output_size = 11

            self.convblock4=nn.Sequential(nn.Conv2d(in_channels=10,out_channels=20,kernel_size=(1,1),padding=0,bias=0),
                             nn.ReLU(),
                             nn.GroupNorm(self.num,20),
                             nn.Dropout(dropout_value)
                             )   # output_size = 11

            self.convblock5=nn.Sequential(nn.Conv2d(in_channels=20,out_channels=10,kernel_size=(3,3),padding=0,bias=0),
                             nn.ReLU(),
                             nn.GroupNorm(self.num,10),
                             nn.Dropout(dropout_value)
                             )  # output_size = 9


            self.convblock6=nn.Sequential(nn.Conv2d(in_channels=10,out_channels=20,kernel_size=(3,3),padding=0,bias=0),
                             nn.ReLU(),
                             nn.GroupNorm(self.num,20),
                             nn.Dropout(dropout_value)
                             ) # output_size = 7
            self.convblock7=nn.Sequential(nn.Conv2d(in_channels=20,out_channels=10,kernel_size=(1,1),padding=0,bias=0),
                             nn.ReLU(),
                             nn.GroupNorm(self.num,10),
                             nn.Dropout(dropout_value)
                             ) # output_size =7

            self.gap = nn.Sequential(
           nn.AvgPool2d(kernel_size=7) # 7>> 9... nn.AdaptiveAvgPool((1, 1))
        ) # output_size = 1
            
            self.convblock8=nn.Sequential(nn.Conv2d(in_channels=10,out_channels=10,kernel_size=(1,1),padding=0,bias=0),
                             
                             ) # output_size =7
        else:
            self.convblock1=nn.Sequential(nn.Conv2d(in_channels=1,out_channels=10,kernel_size=(3,3),padding=0,bias=0),
                             nn.ReLU(),
                             nn.LayerNorm([10, 26, 26]),
                             nn.Dropout(dropout_value)

                             ) # output_size = 26
            self.convblock2=nn.Sequential(nn.Conv2d(in_channels=10,out_channels=20,kernel_size=(3,3),padding=0,bias=0),
                             nn.ReLU(),
                             nn.LayerNorm([20, 24, 24]),
                             nn.Dropout(dropout_value)
                             ) # output_size = 24
            self.convblock3=nn.Sequential(nn.Conv2d(in_channels=20,out_channels=10,kernel_size=(3,3),padding=0,bias=0),
                             nn.ReLU(),
                             nn.LayerNorm([10, 22, 22]),
                             nn.Dropout(dropout_value)
                             )  # output_size = 22


            self.pool1=nn.MaxPool2d(2,2) # output_size = 11

            self.convblock4=nn.Sequential(nn.Conv2d(in_channels=10,out_channels=20,kernel_size=(1,1),padding=0,bias=0),
                             nn.ReLU(),
                             nn.LayerNorm([20, 11, 11]),
                             nn.Dropout(dropout_value)
                             )   # output_size = 11

            self.convblock5=nn.Sequential(nn.Conv2d(in_channels=20,out_channels=10,kernel_size=(3,3),padding=0,bias=0),
                             nn.ReLU(),
                             nn.LayerNorm([10, 9, 9]),
                             nn.Dropout(dropout_value)
                             )  # output_size = 9


            self.convblock6=nn.Sequential(nn.Conv2d(in_channels=10,out_channels=20,kernel_size=(3,3),padding=0,bias=0),
                             nn.ReLU(),
                             nn.LayerNorm([20, 7, 7]),
                             nn.Dropout(dropout_value)
                             ) # output_size = 7
            self.convblock7=nn.Sequential(nn.Conv2d(in_channels=20,out_channels=10,kernel_size=(1,1),padding=0,bias=0),
                             nn.ReLU(),
                             nn.LayerNorm([10, 7, 7]),
                             nn.Dropout(dropout_value)
                             ) # output_size =7

            self.gap = nn.Sequential(
           nn.AvgPool2d(kernel_size=7) # 7>> 9... nn.AdaptiveAvgPool((1, 1))
        ) # output_size = 1
            self.convblock8=nn.Sequential(nn.Conv2d(in_channels=10,out_channels=10,kernel_size=(1,1),padding=0,bias=0),
                             
                             ) # output_size =7
    def forward(self,x):
        x=self.convblock1(x)
        x=self.convblock2(x)
        x=self.convblock3(x)
        x=self.pool1(x)
        x=self.convblock4(x)
        x=self.convblock5(x)
        x=self.convblock6(x)
        x=self.convblock7(x)
        x = self.gap(x)
        x=self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)           
        
    
 


  
    
    



