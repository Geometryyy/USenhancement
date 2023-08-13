# coding: utf-8
# PID : 20829 NCCL_P2P_DISABLE
import os
import datetime
import time
from collections import namedtuple
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from models import *
from myutils import *

cfg: namedtuple
model_path: str
writer: SummaryWriter
classifier: Model
crit: nn.CrossEntropyLoss
optimizer: torch.optim.Adam
lrSche: StepLR
trainLoader: DataLoader
valLoader: DataLoader
scaler: GradScaler


def config():
    global cfg, model_path, writer
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # Configuration
    cfg = {
        'epoch': 0,  # epoch to start training from
        'n_epochs': 400,  # number of epochs of training
        'dataset_name': 'US',  # name of the dataset
        'data_path': '/data/chenliuji/GAN/data/train_datasets',
        'batch_size': 48,  # size of the batches
        'lr': 2e-4,
        'b1':0.5, # adam: decay of first order momentum of gradient, no momentum when using wasserstein loss
        'b2':0.999, # adam: decay of first order momentum of gradient
        'n_cpu': 128,  # number of cpu threads to use during batch generation
        'image_size': [256, 256],
        'channels': 1,
        'checkpoint_interval': 10,  # interval between saving model checkpoints
        'device': [0]
    }
    cfg = namedtuple("Configuration", cfg.keys())(*cfg.values())
    # Create directories
    local_time = time.localtime(time.time())
    time_info = '' if cfg.epoch != 0 else "{}-{}-{}-{}-{}".format(local_time.tm_year,
                                                                  local_time.tm_mon,
                                                                  local_time.tm_mday,
                                                                  local_time.tm_hour,
                                                                  local_time.tm_min)
    model_path = f"/data/chenliuji/GAN/code/USenhancement/AdaIN/{cfg.dataset_name}/{time_info}"
    os.makedirs(model_path + "/models", exist_ok=True)
    writer = SummaryWriter(model_path + '/tensorboard')
    print(cfg)


# make models, criterions, optimizers, lr_schedulers and dataLoaders
def makeVars():
    global cfg, model_path, writer, classifier, crit, optimizer, lrSche, trainLoader, valLoader, scaler

    # Loss
    crit = nn.CrossEntropyLoss().cuda()

    if cfg.epoch != 0:
        classifier = torch.load('/data/chenliuji/GAN/code/USenhancement/AdaIN/US/2023-8-1-23-51/models/vgg_30.pth')
        optimizer = torch.load('/data/chenliuji/GAN/code/USenhancement/AdaIN/US/2023-8-1-23-51/models/opt_30.pth')
        lrSche = torch.load('/data/chenliuji/GAN/code/USenhancement/AdaIN/US/2023-8-1-23-51/models/lrSche_30.pth')
    else:
        classifier = Model(ResNet(BasicBlock, [2, 3, 3, 3, 3, 3]), device=cfg.device).cuda()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))
        lrSche = StepLR(optimizer, step_size=20, gamma=0.9)
    
    scaler = GradScaler()

    # print number of params
    params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print('classifier params:{:,}'.format(params))

    # Datasets and Dataloaders
    Limgs, Himgs = make_Cdataset(cfg.data_path)
    train_L, val_L = train_test_split(Limgs, test_size=0.2, random_state=317)
    train_H, val_H = train_test_split(Himgs, test_size=0.2, random_state=317)
    trainSet = CDataset(train_L, train_H)
    valSet = CDataset(val_L, val_H)
    trainLoader = DataLoader(trainSet, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.n_cpu)
    valLoader = DataLoader(valSet, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.n_cpu)


def train():
    global cfg, model_path, writer, classifier, crit, optimizer, lrSche, trainLoader, valLoader, scaler

    prev_time = time.time()
    time_left = 'need more time to estimate...'
    torch.backends.cudnn.benchmark = True
    len_train = len(trainLoader)
    len_val = len(valLoader)
    for epoch in range(cfg.epoch, cfg.n_epochs + 1):
        # train
        classifier.train()
        total_loss, total_acc= 0, 0
        for i, batch in enumerate(trainLoader):
            Limg, Himg, Y= batch['Limg'].cuda(), batch['Himg'].cuda(), batch['class'].cuda()
            with autocast():
                LY, HY = classifier(Limg), classifier(Himg)
                loss = (crit(LY, Y) + crit(HY, Y)) / 2
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            acc = ((torch.argmax(LY, dim=1) == Y).sum() + (torch.argmax(HY, dim=1) == Y).sum()) / (len(Y) * 2)
            total_loss += loss.item() / len_train
            total_acc += acc.item() / len_train
            
            # print log
            print(
                "\r[Train][Epoch %d/%d][Batch %d/%d][loss: %.4f  acc: %.4f]ETA: %s"
                % (
                    epoch + 1,
                    cfg.n_epochs,
                    i + 1,
                    len_train,
                    loss.item(),
                    acc.item(),
                    time_left
                )
            )

        # tensorboard
        writer.add_scalars('Loss/train', {'Loss': total_loss}, epoch)
        writer.add_scalars('Acc/train', {'Acc': total_acc}, epoch)
        writer.close()


        # Update learning rates
        lrSche.step()

        # val
        classifier.eval()
        total_loss, total_acc= 0, 0
        with torch.no_grad():
            for i, batch in enumerate(valLoader):
                Limg, Himg, Y= batch['Limg'].cuda(), batch['Himg'].cuda(), batch['class'].cuda()
                with autocast():
                    LY, HY = classifier(Limg), classifier(Himg)
                    loss = (crit(LY, Y) + crit(HY, Y)) / 2
                
                acc = ((torch.argmax(LY, dim=1) == Y).sum() + (torch.argmax(HY, dim=1) == Y).sum()) / (len(Y) * 2)
                total_loss += loss.item() / len_val
                total_acc += acc.item() / len_val
                
                # print log
                print(
                    "\r[Val][Epoch %d/%d][Batch %d/%d][loss: %.4f  acc: %.4f]ETA: %s"
                    % (
                        epoch + 1,
                        cfg.n_epochs,
                        i + 1,
                        len_val,
                        loss.item(),
                        acc.item(),
                        time_left
                    )
                )

            # tensorboard
            writer.add_scalars('Loss/val', {'Loss': total_loss}, epoch)
            writer.add_scalars('Acc/val', {'Acc': total_acc}, epoch)
            writer.close()

        # Determine approximate time left
        time_left = str(datetime.timedelta(seconds=int((cfg.n_epochs - epoch) * (time.time() - prev_time))))
        prev_time = time.time()

        if epoch >= 30:
            torch.save(classifier, model_path + f"/models/classifier_{epoch}.pth")
            torch.save(optimizer, model_path + f"/models/opt_{epoch}.pth")
            torch.save(lrSche, model_path + f"/models/lrSche_{epoch}.pth")

def main():
    config()
    makeVars()
    train()


if __name__ == "__main__":
    main()
