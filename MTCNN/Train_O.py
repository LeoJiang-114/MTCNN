#from MTCNN.MTCNN_Net import O_Net
from MTCNN.MTCNN_Net import O_Net
from MTCNN.MTCNN_Trainer import Train

if __name__ == '__main__':

    trainer=Train(net=O_Net(),save_path=r"F:\jkl\MTCNN\Net_Save\O_all.pth",data_path=r"C:\MTCNN_Datasets\48",epoch=150)
    train=trainer.train()
