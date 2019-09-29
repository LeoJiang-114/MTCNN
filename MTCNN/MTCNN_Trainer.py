from .CelebA_Datasets import C_Datasets
import torch
import torch.nn as nn
import torch.utils.data as data
import os
import traceback

class Train():
    def __init__(self,net,data_path,save_path,epoch):
        self.net=net
        self.data_path=data_path
        self.save_path=save_path
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.epoch=epoch

    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net=self.net.to(device)

        data_get=C_Datasets(self.data_path)
        train_datasets=data.DataLoader(dataset=data_get,batch_size=256,shuffle=True,num_workers=2,drop_last=True)

        if os.path.exists(self.save_path):
            self.net=torch.load(self.save_path)#,map_location="cpu"

        cond_loss=nn.BCELoss()
        off_loss=nn.MSELoss()

        for epoch in range(self.epoch):
            for i, (img_data, img_conf, img_off_box) in enumerate(train_datasets):
                img_data, img_off_box, img_conf = img_data.to(device), img_off_box.to(device), img_conf.to(device)
                # print(img_data.shape,img_off_box.shape,img_conf.shape)
                out_conf, out_off = self.net(img_data)
                out_off, out_conf = out_off.view(-1, 4), out_conf.view(-1, 1)
                print(out_off.shape, out_conf.shape)

                img_conf_idxes = img_conf[:, 0] < 2
                # print(img_conf_idxes.shape)
                img_conf_select = img_conf[img_conf_idxes]
                out_conf_select = out_conf[img_conf_idxes]
                print(img_conf_select.shape, out_conf_select.shape)

                loss_conf = cond_loss(out_conf_select, img_conf_select)

                img_off_idxes = img_conf[:, 0] > 0
                img_off_select = img_off_box[img_off_idxes]
                out_off_select = out_off[img_off_idxes]
                print(img_off_select.shape, out_off_select.shape)

                loss_off = off_loss(out_off_select, img_off_select)

                # img_conf_judge = torch.lt(img_conf, 2)
                # img_conf_select = torch.masked_select(img_conf, img_conf_judge)
                # img_conf_select = img_conf_select.view(img_conf_select.shape[0], 1)
                # out_conf_select = torch.masked_select(out_conf, img_conf_judge)
                # out_conf_select=out_conf_select.view(img_conf_select.shape[0], 1)
                # print(img_conf_select.shape,out_conf_select.shape)
                # img_conf_judge = torch.gt(img_conf, 0)
                # img_off_select = torch.masked_select(img_off_box, img_conf_judge)
                # img_off_select = img_off_select.view(int(img_off_select.shape[0] / 4), 4)
                # out_off_select = torch.masked_select(out_off, img_conf_judge)
                # out_off_select = out_off_select.view(int(out_off_select.shape[0] / 4), 4)
                # print(img_off_select.shape)
                # print(out_off_select.shape,img_off_select.shape)

                loss = loss_conf + loss_off

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print("Epoch：{0} -- Loss:{1} -- offLoss:{2} -- confLoss:{3}".format(
                    epoch, loss.cpu().float(), loss_off.cpu().float(), loss_conf.cpu().float()))
            torch.save(self.net, self.save_path)
            print("Save Successful!")