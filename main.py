import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from train_zebra import AdvGAN_Attack #*Using this for changes
from models.model import Target
from models.p2p import Pix2PixModel
from models.options import Options



device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
target = Pix2PixModel(device)
#target.load_state_dict(torch.load('model/gen/gen.mod'))

model_path = 'facades/latest_net_G.pth'
dataset_path = 'facades/image/'
##Modifications for horse/zebra
target.load_state_dict(torch.load(model_path))
dataset = horse2zebraDataset(dataset_path)
train_loader = data.DataLoader(dataset , batch_size=64, shuffle=False, num_workers=4, drop_last=True)


target.cuda()

advGAN = AdvGAN_Attack(device, target, 1)

#advGAN.train(range(10000), 10)
advGAN.train(train_loader, 10)
