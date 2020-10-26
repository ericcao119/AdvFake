import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from train import AdvGAN_Attack
from models.model import Target


device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
target = Target(device)
target.load_state_dict(torch.load('model/gen/gen.mod'))
target.cuda()

advGAN = AdvGAN_Attack(device, target, 1)

advGAN.train(range(10000), 60)