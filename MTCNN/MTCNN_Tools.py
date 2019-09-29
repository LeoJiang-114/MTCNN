import torch
import numpy as np
from PIL import Image
import os
import random

def Pyramid(img):
    w,h=img.size
    img_list=[]
    scale=1
    min_silen=min(w,h)

    while min_silen>12:
        img_list.append([img,scale])
        _w, _h = int(w * scale), int(h * scale)
        img=img.resize((_w,_h))
        scale *= 0.7
        min_silen=min(_w,_h)
    return img_list

def Trans_shpae(img):
    img_data=np.array(img)/255-0.5
    img_data=np.transpose(img_data,(2,0,1))
    img_data=torch.Tensor(img_data)
    return img_data

def IOU_forImg(First_box,Backward_boxes,integrate=False):
    First_Area=(First_box[2]-First_box[0])*(First_box[3]-First_box[1])
    Backward_Area=(Backward_boxes[2]-Backward_boxes[0])*(Backward_boxes[3]-Backward_boxes[1])
    inter_x1 = np.maximum(First_box[0], Backward_boxes[0])
    inter_y1 = np.maximum(First_box[1], Backward_boxes[1])
    inter_x2 = np.minimum(First_box[2], Backward_boxes[2])
    inter_y2 = np.minimum(First_box[3], Backward_boxes[3])
    inter_Area=np.maximum(0,inter_x2-inter_x1)*np.maximum(0,inter_y2-inter_y1)

    if integrate:
        IOU_Result=inter_Area/np.minimum(First_Area,Backward_Area)
    else:
        IOU_Result=inter_Area/(First_Area+Backward_Area-inter_Area)
    return IOU_Result

def IOU_forNMS(First_box,Backward_boxes,integrate=False):
    First_Area=(First_box[2]-First_box[0])*(First_box[3]-First_box[1])
    Backward_Area=(Backward_boxes[:,2]-Backward_boxes[:,0])*(Backward_boxes[:,3]-Backward_boxes[:,1])
    inter_x1 = np.maximum(First_box[0], Backward_boxes[:,0])
    inter_y1 = np.maximum(First_box[1], Backward_boxes[:,1])
    inter_x2 = np.minimum(First_box[2], Backward_boxes[:,2])
    inter_y2 = np.minimum(First_box[3], Backward_boxes[:,3])
    inter_Area=np.maximum(0,inter_x2-inter_x1)*np.maximum(0,inter_y2-inter_y1)

    if integrate:
        IOU_Result=np.true_divide(inter_Area,np.minimum(First_Area,Backward_Area))
    else:
        IOU_Result=np.true_divide(inter_Area,(First_Area+Backward_Area-inter_Area))
    return IOU_Result

def NMS(boxes,conf_judge,integrate=False):
    if boxes.shape[0] == 0:
        return np.array([])
    index=(-boxes[:,4]).argsort()
    _boxes=boxes[index]
    save_boxes=[]

    while _boxes.shape[0]>1:
        First_box=_boxes[0]
        Backward_boxes=_boxes[1:]
        save_boxes.append(First_box)
        IOU_Result=IOU_forNMS(First_box,Backward_boxes,integrate)
        index=np.where(IOU_Result<conf_judge)
        _boxes=Backward_boxes[index]
    if _boxes.shape[0]>0:
        save_boxes.append(_boxes[0])
    save_boxes=np.stack(save_boxes)
    return save_boxes

def To_Square(boxes):
    boxes_square=boxes.copy()
    w=boxes[:,2]-boxes[:,0]
    h=boxes[:,3]-boxes[:,1]

    max_silen=np.maximum(w,h)

    boxes_square[:, 0] = boxes[:, 0] - w / 2 + max_silen / 2
    boxes_square[:, 1] = boxes[:, 1] - h / 2 + max_silen / 2
    boxes_square[:, 2] = boxes_square[:, 0] + max_silen
    boxes_square[:, 3] = boxes_square[:, 1]  + max_silen
    return boxes_square

def Crop_Resize_Img(img,boxes,size):
    crop_list=[]
    xy_list=[]
    for line in boxes:
        x1=line[0]
        y1=line[1]
        x2=line[2]
        y2=line[3]
        xy_list.append([x1,y1,x2,y2])
        img_crop=img.crop((x1,y1,x2,y2))
        img_resize=img_crop.resize((size,size))
        # img_data=Trans_shpae(img_resize)
        crop_list.append(img_resize)
    return crop_list,np.array(xy_list)


if __name__ == '__main__':
    path=r"E:\标注完的\img\000001.jpg"
    img_open=Image.open(path)
    # pic_py=Pyramid(img)
    # print(pic_py)
    # for i in pic_py:
    #     print(i[0].size)

    # img_data=Trans_shpae(img)
    # img_data=torch.unsqueeze(img_data,0)
    # print(img_data,img_data.shape)

    boxes=np.array([[184, 276, 238, 400],
                [156, 240, 207, 400],
                [198, 266, 232, 400],
                [125, 228, 235, 450]])
    boxes=np.array(boxes)
    boxes=To_Square(boxes)
    img_crop,xy_list=Crop_Resize_Img(img_open,boxes,24)
    print(xy_list)
    # img_data=[]
    # for img in img_crop:
    #     img_trans = Trans_shpae(img)
    #     img_data.append(img_trans)
    # img_data=torch.stack(img_data)
    # print(img_data.shape)
    # for _img in img_crop:
    #     _img.show()
