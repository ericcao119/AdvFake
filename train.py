import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np

import torchvision
from torchvision import datasets
import os
import random
from os import listdir
from os.path import isfile, join


from models.generator import Generator
from models.discriminator import Discriminator


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = original_tuple + (path,)

        # TODO Add alignments here
        return tuple_with_path


# instantiate the dataset and dataloader
data_dir = "your/data_dir/here"
train_dataset = ImageFolderWithPaths(root="./data/train")  # our custom dataset
dataloader = DataLoader(train_dataset, batch_size=64, num_workers=1, shuffle=True)

test_dataset = torchvision.ImageFolderWithPaths(
    root="./data/test", transform=torchvision.transforms.ToTensor()
)

models_path = "./models/instances"


def swapfaces(original_image, base_image):
    """TODO create temporary folder for single input image. One for extracted faces, one for the image to patch, and one for the new complete output
    Apply face to base_image with extract and convert from model
    """


def cropped_image(original_image):
    pass


def random_image(face):
    def training_set():
        # Get training data in sorted filepath form
        train_path = "./data/train"
        data = sorted([f for f in listdir(train_path) if isfile(join(train_path, f))])
        return data

    return random.choice(training_set())


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.xavier_uniform_(m.weight.data, 0.05)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class AdvGAN_Attack:
    def __init__(self, device, model, image_nc, box_min, box_max):
        output_nc = image_nc
        self.device = device
        self.model = model
        self.input_nc = image_nc
        self.output_nc = output_nc

        self.box_min = box_min
        self.box_max = box_max

        self.gen_input_nc = image_nc
        self.netG = Generator(self.gen_input_nc, image_nc).to(device)
        self.netDisc = Discriminator(image_nc).to(device)

        # initialize all weights
        self.netG.apply(weights_init)
        self.netDisc.apply(weights_init)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=0.001)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(), lr=0.001)

        if not os.path.exists(models_path):
            os.makedirs(models_path)

    def train_batch(self, original_image):
        """x is the large not cropped face. TODO find a way to associate image with the image it came from (see if we can do it by filename)"""
        x = cropped_image(original_image)  # TODO Crop a 256 x 256 square from the image

        # optimize D
        perturbation = self.netG(x)
        base_image = random_image()  # Image to apply face to

        # add a clipping trick
        adv_images = torch.clamp(perturbation, -0.3, 0.3) + x
        adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

        for i in range(1):
            self.optimizer_D.zero_grad()
            pred_real = self.netDisc(x)
            loss_D_real = F.mse_loss(
                pred_real, torch.ones_like(pred_real, device=self.device)
            )
            loss_D_real.backward()

            pred_fake = self.netDisc(adv_images.detach())
            loss_D_fake = F.mse_loss(
                pred_fake, torch.zeros_like(pred_fake, device=self.device)
            )
            loss_D_fake.backward()
            loss_D_GAN = loss_D_fake + loss_D_real
            self.optimizer_D.step()

        # optimize G
        for i in range(1):
            self.optimizer_G.zero_grad()

            # cal G's loss in GAN
            pred_fake = self.netDisc(adv_images)
            loss_G_fake = F.mse_loss(
                pred_fake, torch.ones_like(pred_fake, device=self.device)
            )
            loss_G_fake.backward(retain_graph=True)

            # calculate perturbation norm
            C = 0.1
            loss_perturb = torch.mean(
                torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1)
            )

            # 1 - image similarity
            # TODO apply image back to original image
            # ex. perturbed_original = (adv_images patched onto the original image)
            # Clamp it
            # perform face swap with the images

            faked_image = swapfaces(adv_images, base_image)
            norm_similarity = torch.abs(
                torch.tensor(1.0)
                - torch.dot(faked_image, base_image) / torch.dot(base_image, base_image)
            )
            loss_adv = norm_similarity

            adv_lambda = 10
            pert_lambda = 1
            loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb
            loss_G.backward()
            self.optimizer_G.step()

        return (
            loss_D_GAN.item(),
            loss_G_fake.item(),
            loss_perturb.item(),
            loss_adv.item(),
        )

    def train(self, train_dataloader, epochs):
        for epoch in range(1, epochs + 1):

            if epoch == 50:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=0.0001)
                self.optimizer_D = torch.optim.Adam(
                    self.netDisc.parameters(), lr=0.0001
                )
            if epoch == 80:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=0.00001)
                self.optimizer_D = torch.optim.Adam(
                    self.netDisc.parameters(), lr=0.00001
                )
            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            for i, data in enumerate(train_dataloader, start=0):
                (images, paths) = data
                images = images.to(self.device)

                (
                    loss_D_batch,
                    loss_G_fake_batch,
                    loss_perturb_batch,
                    loss_adv_batch,
                ) = self.train_batch(images)
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch

            # print statistics
            num_batch = len(train_dataloader)
            print(
                "epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f,\
             \nloss_perturb: %.3f, loss_adv: %.3f, \n"
                % (
                    epoch,
                    loss_D_sum / num_batch,
                    loss_G_fake_sum / num_batch,
                    loss_perturb_sum / num_batch,
                    loss_adv_sum / num_batch,
                )
            )

            # save generator
            if epoch % 20 == 0:
                netG_file_name = models_path + "netG_epoch_" + str(epoch) + ".pth"
                torch.save(self.netG.state_dict(), netG_file_name)