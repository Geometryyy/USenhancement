import torch
import PIL.Image as Image
from models import *
from myutils import *
# tfs = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=0.5, std=0.5, inplace=True)])
# x = tfs(Image.open('/data/chenliuji/GAN/data/low_quality_images/0970.png')).unsqueeze(0)
# _, Himgs = make_dataset('/data/chenliuji/GAN/data/train_datasets')
# max = {'id':0, 'value':0}
# for Himg in Himgs:
#     index = ssim(x, tfs(Image.open(Himg)).unsqueeze(0))
#     if index > max['value']:
#         max['value'] = index
#         max['id'] = Himg
# print(max)