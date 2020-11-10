import torch

class RandomNoise:

    def __init__(self, target, box_min=0, box_max=1):
        self.target = target
        self.box_min = box_min
        self.box_max = box_max

    def random_perturb(self, x, max_perturb=0.3, max_norm=0.1):
        """x is the large not cropped face. TODO find a way to associate image with the image it came from (see if we can do it by filename)"""
        # x is the cropped 256x256 to perturb
        # optimize D
        x_image = x.reshape((1, 1, 10, 10))

        # Generate perturbation and restrict the max perturbation to part of a pixel
        perturbation = torch.rand(1, 1, 10, 10)
        perturbation *= max_perturb
        
        # Restrict the total norm of the vector
        norm = torch.norm(perturbation, 2)
        perturbation *= (max_norm / norm)

        # Make the image valid
        protected_image = perturbation + x_image

        # apply the adversarial image
        perturbed_output = self.target(protected_image)

        return protected_image, perturbed_output