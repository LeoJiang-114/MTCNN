import torch
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channels,in_channels//2,1,1,0),
            nn.Conv2d(in_channels//2,in_channels,3,1,1)
        )

    def forward(self, x):
        output=x+self.layer(x)
        return output

class P_Net(nn.Module):
    def __init__(self):
        super(P_Net,self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(3,10,3,1),
            nn.PReLU(),
            nn.MaxPool2d(2,2),
            ResNet(10),
            nn.Conv2d(10,16,3,1),
            nn.PReLU(),
            ResNet(16),
            nn.Conv2d(16,32,3,1),
            nn.PReLU()
        )
        self.confidence=nn.Conv2d(32,1,1,1)
        self.sigmoid=nn.Sigmoid()
        self.offset=nn.Conv2d(32,4,1,1)

    def forward(self, x):
        input=self.layer(x)

        # print(output_conf)
        output_conf=self.sigmoid(self.confidence(input))

        output_off=self.offset(input)
        return output_conf,output_off

class R_Net(nn.Module):
    def __init__(self):
        super(R_Net, self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(3,28,3,1,1),
            nn.PReLU(),
            nn.MaxPool2d(3,2),
            ResNet(28),
            nn.Conv2d(28,48,3,1),
            nn.PReLU(),
            nn.MaxPool2d(3,2),
            ResNet(48),
            nn.Conv2d(48,64,2,1),
            nn.PReLU()
        )
        self.layer2=nn.Sequential(
            nn.Linear(3*3*64, 128),
            nn.PReLU()
        )
        self.confidence = nn.Linear(128, 1)
        self.sigmoid=nn.Sigmoid()
        self.offset = nn.Linear(128, 4)

    def forward(self, x):
        input=self.layer1(x)
        input=input.view(-1,3*3*64)
        input=self.layer2(input)

        output_conf=self.confidence(input)
        output_conf=self.sigmoid(output_conf)

        output_off=self.offset(input)
        return output_conf,output_off

class O_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(3,32,3,1,1),
            nn.PReLU(),
            nn.MaxPool2d(3,2),
            ResNet(32),
            nn.Conv2d(32,64,3,1),
            nn.PReLU(),
            nn.MaxPool2d(3,2),
            ResNet(64),
            nn.Conv2d(64,64,3,1),
            nn.PReLU(),
            nn.MaxPool2d(2,2),
            ResNet(64),
            nn.Conv2d(64,128,2,1),
            nn.PReLU(),
        )
        self.layer2=nn.Sequential(
            nn.Linear(3*3*128,256),
            nn.PReLU(),
        )
        self.confidence=nn.Linear(256,1)
        self.sigmoid=nn.Sigmoid()
        self.offset=nn.Linear(256,4)

    def forward(self, x):
        input = self.layer1(x)
        input = input.view(-1, 3 * 3 * 128)
        input = self.layer2(input)

        output_conf=self.confidence(input)
        #print(output_conf)
        output_conf=self.sigmoid(output_conf)

        output_offset=self.offset(input)
        return output_conf,output_offset

if __name__ == '__main__':
    rnet=P_Net()
    array=torch.randn(2,3,144,144)
    conf,off=rnet(array)
    print(conf.shape,off)