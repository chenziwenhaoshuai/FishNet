import torch
from torch import nn
import torchvision


class ConvolutionalLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvolutionalLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU()
        )

    def forward(self, x):
        return self.sub_module(x)


class ConvolutionalSet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionalSet, self).__init__()

        self.sub_module = torch.nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1),

            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1),

            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        return self.sub_module(x)


class mymodel(torch.nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        model1 = torchvision.models.resnet18(pretrained=True)
        conv1 = model1.conv1
        bn1 = model1.bn1
        relu = model1.relu
        maxpool = model1.maxpool
        layer1 = model1.layer1
        layer2 = model1.layer2
        layer3 = model1.layer3
        layer4 = model1.layer4

        backbone = torch.nn.Sequential(conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4)
        self.convset_13 = torch.nn.Sequential(
            ConvolutionalSet(512, 512)
        )
        self.detetion_13 = torch.nn.Sequential(
            ConvolutionalLayer(512, 1024, 3, 1, 1),
            torch.nn.Conv2d(1024, 270, 1, 1, 0)
        )
        self.backbone = backbone

    def forward(self,x):
        backbone = self.backbone(x) # [1, 512, 20, 20]
        convset_out_13 = self.convset_13(backbone)
        detetion_out_13 = self.detetion_13(convset_out_13)
        return detetion_out_13


if __name__ == '__main__':
    mymodel = mymodel()
    image  = torch.randn(5,3,416,416)
    pred = mymodel(image)
    print(pred.shape)