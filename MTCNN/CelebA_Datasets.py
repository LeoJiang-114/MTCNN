import torch.utils.data as data
import torch
import numpy as np
import os
from PIL import Image,ImageDraw

class C_Datasets(data.Dataset):
    def __init__(self,path):
        super().__init__()
        self.path=path
        self.data_lines=[]
        self.data_lines.extend(open(os.path.join(self.path,"positive.txt"),"r").readlines())
        self.data_lines.extend(open(os.path.join(self.path,"part.txt"),"r").readlines())
        self.data_lines.extend(open(os.path.join(self.path,"negative.txt"),"r").readlines())

    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, index):
        line=self.data_lines[index].split()
        img_path=line[0]
        img_conf=torch.Tensor([int(line[1])])
        img_off_box=torch.Tensor([float(line[2]),float(line[3]),float(line[4]),float(line[5])])
        img_open=Image.open(os.path.join(self.path,img_path))
        img_data=np.transpose(np.array(img_open),(2,0,1))
        img_data=torch.Tensor(img_data/255-0.5)

        return img_data,img_conf,img_off_box

if __name__ == '__main__':
    path=r"C:\MTCNN_Datasets\12"
    datasets=C_Datasets(path)
    img_data=datasets[0][0]
    img_conf = datasets[0][1]
    img_off=datasets[0][2]

    print(img_data.shape)
    print(img_off,img_conf)

    # img_path=r"F:\Datasets_Original\标注完的_CelebA\img\000005.jpg"
    # img_open=Image.open(img_path)
    # img_draw=ImageDraw.Draw(img_open)
    # img_draw.rectangle((int(img_off[0]*120+236),int(img_off[1]*166+109),int(img_off[2]*120+236+120),int(img_off[3]*166+109+166)),outline="red")
    # img_draw.rectangle((236,109,236+120,109+166),outline="green")
    # print(int(img_off[0]*120+236),int(img_off[1]*166+109),int(img_off[2]*120+236+120),int(img_off[3]*166+109+166))
    # img_open.show()