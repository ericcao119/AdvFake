from pathlib import Path
from config import FACESWAP_PATH

from torch import nn

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

class Target(nn.Module):
    def __init__(self, ngpu):
        super(Target, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# import subprocess

# class Model:

#     def __init__(self):
#         pass

#     def center_box(self, row, col, image):
#         pass

#     def extract(self, input_dir, output_dir):
#         """Extracts faces from photos"""
#         subprocess.run([
#             "python",
#             FACESWAP_PATH,
#             "extract",
#             "-i",
#             input_dir,
#             "-o",
#             output_dir
#         ])
        
    
#     def convert(self, input_dir, output_dir):
#         """Swaps the faces from the input photos to the output photos"""
#         subprocess.run([
#             "python",
#             FACESWAP_PATH,
#             "convert",
#             "-i",
#             input_dir,
#             "-o",
#             output_dir
#         ])

