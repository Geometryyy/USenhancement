# coding: utf-8
# PID : 20829 NCCL_P2P_DISABLE
import os
import datetime
import time
from collections import namedtuple
from sklearn.model_selection import train_test_split
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from models import *
from myutils import *

cfg: namedtuple
model_path: str
writer: SummaryWriter
model: nn.Module
optimizer: torch.optim.Adam
lrSche: StepLR
trainLoader: DataLoader
valLoader: DataLoader
scaler: GradScaler


def sample_images(epoch):
    global cfg, valLoader, model
    imgs = next(iter(valLoader))
    model.eval()
    real_A, real_B = imgs['Limg'][:4].cuda(), imgs['Himg'][:4].cuda()
    fake_B = model.generate(real_A, real_B)

    # Arrange images along x-axis
    real_A = make_grid(real_A, nrow=4, normalize=True)
    real_B = make_grid(real_B, nrow=4, normalize=True)
    fake_B = make_grid(fake_B, nrow=4, normalize=True)

    # Arrange images along y-axis
    image_grid = torch.cat((real_A, real_B, fake_B), 1)
    for i in range(cfg.channels):
        save_image(image_grid[i], model_path + f"/images/{epoch}_{i}.png")


def config():
    global cfg, model_path, writer
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
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
        'device': [0, 1]
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
    os.makedirs(model_path + "/images", exist_ok=True)
    os.makedirs(model_path + "/models", exist_ok=True)
    writer = SummaryWriter(model_path + '/tensorboard')
    print(cfg)


# make models, criterions, optimizers, lr_schedulers and dataLoaders
def makeVars():
    global cfg, model_path, writer, model, optimizer, lrSche, trainLoader, valLoader, scaler
    if cfg.epoch != 0:
        model = torch.load('')
        optimizer = torch.load('')
        lrSche = torch.load('')
    else:
        model = AdaINModel().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))
        lrSche = StepLR(optimizer, step_size=20, gamma=0.9)
    
    scaler = GradScaler()

    # print number of params
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model params:{:,}'.format(params))

    # Datasets and Dataloaders
    Limgs, Himgs = make_dataset(cfg.data_path)
    train_L, val_L = train_test_split(Limgs, test_size=0.2, random_state=317)
    train_H, val_H = train_test_split(Himgs, test_size=0.2, random_state=317)
    trainSet = USDataset(train_L, train_H)
    valSet = USDataset(val_L, val_H)
    trainLoader = DataLoader(trainSet, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.n_cpu)
    valLoader = DataLoader(valSet, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.n_cpu)


def train():
    global cfg, model_path, writer, model, optimizer, lrSche, trainLoader, valLoader, scaler

    prev_time = time.time()
    time_left = 'need more time to estimate...'
    torch.backends.cudnn.benchmark = True
    len_train = len(trainLoader)
    len_val = len(valLoader)
    for epoch in range(cfg.epoch, cfg.n_epochs + 1):
        # train
        model.train()
        total_loss, total_PSNR, total_SSIM, total_LNCC= 0, 0, 0, 0
        for i, batch in enumerate(trainLoader):
            Limg, Himg= batch['Limg'].cuda(), batch['Himg'].cuda()
            with autocast():
                loss, fake_H = model(Limg, Himg)
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # compute indicators
            fake_H = fake_H.float()
            PSNR = 10 * torch.log10(4 / ((Himg - fake_H) ** 2).mean())
            SSIM = ssim(fake_H, Himg)
            LNCC = lncc(fake_H, Himg)
            
            total_loss += loss.item() / len_train
            total_PSNR += PSNR.item() / len_train
            total_SSIM += SSIM.item() / len_train
            total_LNCC += LNCC.item() / len_train
            
            # print log
            print(
                "\r[Train][Epoch %d/%d][Batch %d/%d][loss: %.4f  PSNR: %.4f  SSIM: %.4f  LNCC: %.4f]ETA: %s"
                % (
                    epoch + 1,
                    cfg.n_epochs,
                    i + 1,
                    len_train,
                    loss,
                    PSNR,
                    SSIM,
                    LNCC,
                    time_left
                )
            )

        # tensorboard
        writer.add_scalars('Loss/train', {'Loss': total_loss}, epoch)
        writer.add_scalars('PSNR/train', {'PSNR': total_PSNR}, epoch)
        writer.add_scalars('SSIM/train', {'SSIM': total_SSIM}, epoch)
        writer.add_scalars('LNCC/train', {'LNCC': total_LNCC}, epoch)
        writer.close()


        # Update learning rates
        lrSche.step()

        # val
        model.eval()
        total_loss, total_PSNR, total_SSIM, total_LNCC= 0, 0, 0, 0
        with torch.no_grad():
            for i, batch in enumerate(valLoader):
                Limg, Himg= batch['Limg'].cuda(), batch['Himg'].cuda()
                with autocast():
                    loss, fake_H = model(Limg, Himg)
                
                fake_H = fake_H.float()
                PSNR = 10 * torch.log10(4 / ((Himg - fake_H) ** 2).mean())
                SSIM = ssim(fake_H, Himg)
                LNCC = lncc(fake_H, Himg)
                
                total_loss += loss.item() / len_val
                total_PSNR += PSNR.item() / len_val
                total_SSIM += SSIM.item() / len_val
                total_LNCC += LNCC.item() / len_val
                
                # print log
                print(
                    "\r[Val][Epoch %d/%d][Batch %d/%d][loss: %.4f  PSNR: %.4f  SSIM: %.4f  LNCC: %.4f]ETA: %s"
                    % (
                        epoch + 1,
                        cfg.n_epochs,
                        i + 1,
                        len_val,
                        loss,
                        PSNR,
                        SSIM,
                        LNCC,
                        time_left
                    )
                )

            # tensorboard
            writer.add_scalars('Loss/val', {'Loss': total_loss}, epoch)
            writer.add_scalars('PSNR/val', {'PSNR': total_PSNR}, epoch)
            writer.add_scalars('SSIM/val', {'SSIM': total_SSIM}, epoch)
            writer.add_scalars('LNCC/val', {'LNCC': total_LNCC}, epoch)
            writer.close()

        # Determine approximate time left
        time_left = str(datetime.timedelta(seconds=int((cfg.n_epochs - epoch) * (time.time() - prev_time))))
        prev_time = time.time()
        sample_images(epoch)
        if epoch >= 30:
            torch.save(model, model_path + f"/models/model_{epoch}.pth")
            torch.save(optimizer, model_path + f"/models/opt_{epoch}.pth")
            torch.save(lrSche, model_path + f"/models/lrSche_{epoch}.pth")
            

def main():
    config()
    makeVars()
    train()


if __name__ == "__main__":
    main()
