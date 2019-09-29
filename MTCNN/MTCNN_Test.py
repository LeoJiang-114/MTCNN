import torch
import numpy as np
import os
from PIL import Image,ImageDraw,ImageFont
import MTCNN.tool.MTCNN_Tools as tools
import time

class Detector:
    def __init__(self,net_path=r"F:\jkl\MTCNN\Net_Save"):
        self.Pnet = torch.load(os.path.join(net_path, "P_long.pth"))
        self.Rnet = torch.load(os.path.join(net_path, "R_long.pth"))
        self.Onet = torch.load(os.path.join(net_path, "O_long.pth"))

        # self.Pnet = torch.load(os.path.join(net_path, "P_all.pth"))
        # self.Rnet = torch.load(os.path.join(net_path, "R_all.pth"))
        # self.Onet = torch.load(os.path.join(net_path, "O_all.pth"))

        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.Pnet = self.Pnet.to(self.device)
        self.Rnet = self.Rnet.to(self.device)
        self.Onet = self.Onet.to(self.device)

    def detector(self,img):
        start_time=time.time()
        P_boxes=[]
        img_pyramid=tools.Pyramid(img)
        for img_get in img_pyramid:
            P_boxes.extend(self.P_Detector(img=img_get[0],scale=img_get[1],conf_judge=0.5))
        if len(P_boxes) == 0:
            print("P没有脸！")
            return np.array([])
        P_boxes=np.stack(P_boxes)
        print("P_boxes:",P_boxes.shape)



        Boxes_Square_P=tools.To_Square(P_boxes)
        ImgCrop_P,boxes_P=tools.Crop_Resize_Img(img,Boxes_Square_P,24)
        R_boxes=self.R_detector(ImgCrop_P=ImgCrop_P,boxes_P=boxes_P,conf_judge=0.5)
        print("R_boxes:", R_boxes.shape)

        if R_boxes.shape[0]==0:
            print("R没有脸！")
            return np.array([])

        Boxes_Square_R = tools.To_Square(R_boxes)
        ImgCrop_R, boxes_R = tools.Crop_Resize_Img(img, Boxes_Square_R, 48)
        O_boxes = self.O_detector( ImgCrop_R=ImgCrop_R, boxes_R=boxes_R,conf_judge=0.5)

        if O_boxes.shape[0] == 0:
            print("O没有脸！")
            return np.array([])

        end_time=time.time()
        print("all_time:",end_time-start_time)

        return O_boxes

    def O_detector(self,conf_judge,ImgCrop_R,boxes_R):
        img_data=[]
        for img_crop in ImgCrop_R:
            img_crop=tools.Trans_shpae(img_crop)
            img_data.append(img_crop)
        img_data=torch.stack(img_data)
        # print(img_data.shape)

        img_data=img_data.to(self.device)
        conf,offset=self.Onet(img_data)
        conf,offset=conf.cpu().data,offset.cpu().data
        # print(conf, offset.shape)

        mask=conf[:,0]>conf_judge
        idxes=mask.nonzero()
        conf=conf[idxes].view(-1,1)
        offset=offset[idxes].view(-1,4)

        boxes_R=torch.Tensor(boxes_R[idxes]).view(-1,4)
        # print(conf.shape,offset.shape,boxes_R.shape)

        w,h=boxes_R[:,2]-boxes_R[:,0],boxes_R[:,3]-boxes_R[:,1]

        R_boxes_=torch.zeros(boxes_R.size(0),boxes_R.size(1)+1)
        R_boxes_[:, 0]=boxes_R[:,0]+offset[:,0]*w
        R_boxes_[:, 1]=boxes_R[:,1]+offset[:,1]*h
        R_boxes_[:, 2]=boxes_R[:,2]+offset[:,2]*w
        R_boxes_[:, 3]=boxes_R[:,3]+offset[:,3]*h
        R_boxes_[:,4]=conf[:,0]
        # print(R_boxes_)
        boxes=R_boxes_.numpy()
        boxes_NMS=tools.NMS(boxes,0.5,True)
        return boxes_NMS

    def R_detector(self,conf_judge,ImgCrop_P,boxes_P):
        img_data=[]
        for img_crop in ImgCrop_P:
            img_crop=tools.Trans_shpae(img_crop)
            img_data.append(img_crop)
        img_data=torch.stack(img_data)
        # print(img_data.shape)
        img_data = img_data.to(self.device)
        conf,offset=self.Rnet(img_data)
        conf,offset=conf.cpu().data,offset.cpu().data
        mask=conf[:,0]>conf_judge
        idxes=mask.nonzero()
        conf=conf[idxes].view(-1,1)
        offset=offset[idxes].view(-1,4)
        boxes_P=torch.Tensor(boxes_P[idxes]).view(-1,4)
        # print(conf.shape,offset.shape,boxes_P.shape)

        w,h=boxes_P[:,2]-boxes_P[:,0],boxes_P[:,3]-boxes_P[:,1]

        R_boxes=torch.zeros(boxes_P.size(0),boxes_P.size(1)+1)
        R_boxes[:, 0]=boxes_P[:,0]+offset[:,0]*w
        R_boxes[:, 1]=boxes_P[:,1]+offset[:,1]*h
        R_boxes[:, 2]=boxes_P[:,2]+offset[:,2]*w
        R_boxes[:, 3]=boxes_P[:,3]+offset[:,3]*h
        R_boxes[:,4]=conf[:,0]

        boxes=R_boxes.numpy()
        boxes_NMS=tools.NMS(boxes,0.5,False)
        return boxes_NMS

    def P_Detector(self,img,scale,conf_judge):
        img_data=tools.Trans_shpae(img)
        img_data=torch.unsqueeze(img_data,0)
        img_data=img_data.to(self.device)
        conf,offset=self.Pnet(img_data)

        #矩阵运算
        conf,offset=conf.cpu().data.numpy(),offset[0].cpu().data.numpy()
        #print(conf.shape,offset.shape)
        idxes=np.where(conf>conf_judge)
        select_conf=conf[idxes]
        i,j=idxes[2],idxes[3]
        _x1 = j * 2 / scale
        _y1 = i * 2 / scale
        _x2 = (j * 2 + 12) / scale
        _y2 = (i * 2 + 12) / scale

        o_w=_x2-_x1
        o_h=_y2-_y1

        x1=offset[0,i,j]*o_w+_x1
        y1=offset[1,i,j]*o_h+_y1
        x2=offset[2,i,j]*o_w+_x2
        y2=offset[3,i,j]*o_h+_y2
        boxes=np.vstack((x1,y1,x2,y2,select_conf))
        boxes=boxes.T

        # conf, offset = conf[0][0].cpu().data.numpy(), offset[0].cpu().data.numpy()
        # boxes=[]
        # for i,conf_0 in enumerate(conf):
        #     for j,conf_1 in enumerate(conf_0):
        #         if conf_1>conf_judge:
        #             _x1 = (j * 2) / scale
        #             _y1 = (i * 2) / scale
        #             _x2 = (j * 2 + 12) / scale
        #             _y2 = (i * 2 + 12) / scale
        #
        #             o_w = _x2 - _x1
        #             o_h = _y2 - _y1
        #
        #             x1 = offset[0, i, j] * o_w + _x1
        #             y1 = offset[1, i, j] * o_h + _y1
        #             x2 = offset[2, i, j] * o_w + _x2
        #             y2 = offset[3, i, j] * o_h + _y2
        #             boxes.append([round(x1), round(y1), round(x2), round(y2), float(conf_1)])
        # boxes=np.array(boxes)

        boxes_NMS=tools.NMS(boxes,0.5,False)
        return boxes_NMS

if __name__ == '__main__':
    img_path=r"F:\Picture_M\Myself\一寸.jpg"
    img_open = Image.open(img_path)
    # img_open=Image.open(r"F:\Picture_M\Myself\20180608_232208_mh1528815229695.jpg")
    img_open = img_open.convert("RGB")
    w, h = img_open.size
    #img_open=img_open.resize((int(w/2),int(h/2)))
    img_draw = ImageDraw.Draw(img_open)
    font=ImageFont.truetype(font=r"F:\jkl\MTCNN\arial.ttf",size=20,)

    detector = Detector()
    boxes = detector.detector(img_open)
    np.set_printoptions(suppress=True)
    print(boxes, boxes.shape)
    for i in boxes:
        x1 = i[0]
        y1 = i[1]
        x2 = i[2]
        y2 = i[3]
        img_draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
        #print(i[4],type(i[4]))
        img_draw.text((x1,y1-20),text="{:.3f}".format(i[4]),fill="red",font=font)
    img_open.show()
