# coding: utf-8
# PID : 20829 NCCL_P2P_DISABLE
import os
from collections import namedtuple
from torchvision.utils import save_image
from models import *
from myutils import *

cfg: namedtuple
model_path: str
G_AB: Model


def config():
    global cfg, model_path, writer
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5'
    # Configuration
    cfg = {
        'dataset_name': 'BAO_MRI',  # name of the dataset
        'data_path': '/data/chenliuji/GAN/data/low_quality_images',
        'n_cpu': 128,  # number of cpu threads to use during batch generation
        'img_width': 256,
        'img_height': 256,
        'channels': 1,
        'device': [0, 1, 2, 3]
    }
    cfg = namedtuple("Configuration", cfg.keys())(*cfg.values())
    # Create directories
    model_path = f"/data/chenliuji/GAN/code/USenhancement/gancopy/US/2023-7-29-13-14"
    os.makedirs(model_path + "/test-images", exist_ok=True)
    print(cfg)


def makeVars():
    global cfg, model_path, G_AB
    
    # models, optimizers and lr_schedulers
    G_AB = torch.load("/data/chenliuji/GAN/code/USenhancement/gancopy/US/2023-7-29-13-14/models/G_AB_220.pth")
        
    # print number of params
    params = sum(p.numel() for p in G_AB.parameters() if p.requires_grad)
    print('G params:{:,}'.format(params))


def main():
    global cfg, G_AB
    config()
    makeVars()
    G_AB.eval()
    img_ids = os.listdir(cfg.data_path)
    tfs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5, inplace=True)
        ])
    with torch.no_grad():
        for img_id in img_ids:
            limg = tfs(Image.open(os.path.join(cfg.data_path, img_id)))
            himg = G_AB(limg.unsqueeze(0))
            save_image(himg, os.path.join(model_path, "test-images", img_id), normalize=True)
            print(img_id)


if __name__ == "__main__":
    main()