#import MTCNN.MTCNN_Net as Net
import MTCNN.MTCNN_Net as Net
import MTCNN.MTCNN_Trainer as Trainer

if __name__ == '__main__':
    net=Net.R_Net()
    trainer=Trainer.Train(net=net,save_path=r"F:\jkl\MTCNN\Net_Save\R_all.pth",data_path=r"C:\MTCNN_Datasets\24",epoch=150)
    train=trainer.train()