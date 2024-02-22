"""
Contains PyTorch model code to instantiate a PCONet model.
"""
import torch
from torch import nn
import math

class PCONet(nn.Module):
    def __init__(self,input_shape,output_shape):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=32,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=0),
        nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.conv_block_3 = nn.Sequential(
        nn.Conv2d(in_channels=32,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=0),
        nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.conv_block_4 = nn.Sequential(
        nn.Conv2d(in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=0),
        nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=3200,
                      out_features=128),
            nn.SiLU(),
            # nn.Dropout(0.5),
            nn.Linear(in_features=128,
                      out_features=256),
            nn.SiLU(),
            # nn.Dropout(0.5),
            nn.Linear(in_features=256,
                      out_features=output_shape)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                torch.nn.init.zeros_(m.bias)
                
    def forward(self,x):
        return self.classifier(self.conv_block_5(self.conv_block_4(self.conv_block_3(self.conv_block_2(self.conv_block_1(x))))))
        