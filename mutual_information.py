import torch
import torch.nn as nn
from icecream import ic


class MutualInformation(nn.Module):
    # Implementation inspired from https://github.com/connorlee77/pytorch-mutual-information
    # They did it on image, here for 2dim embedding

    def __init__(self, sigma=0.4, normalize=True):
        super(MutualInformation, self).__init__()

        self.sigma = 2 * sigma ** 2
        self.normalize = normalize
        self.epsilon = 1e-10

    def marginalPdf(self, values):
        kernel_values = torch.exp(-0.5 * (values / self.sigma).pow(2))
        pdf = torch.mean(kernel_values, dim=1)

        return pdf, kernel_values

    def jointPdf(self, kernel_values1, kernel_values2):
        joint_kernel_values = torch.mul(kernel_values1, kernel_values2)
        normalization = torch.sum(joint_kernel_values, dim=1).view(-1, 1) + self.epsilon
        pdf = joint_kernel_values / normalization

        return pdf

    def getMutualInformation(self, input1, input2):
        """
            input1: B, S1
            input2: B, S2
            return: scalar
        """

        pdf_x1, kernel_values1 = self.marginalPdf(input1)
        pdf_x2, kernel_values2 = self.marginalPdf(input2)
        pdf_x1x2 = self.jointPdf(kernel_values1, kernel_values2)

        H_x1 = -pdf_x1 * torch.log2(pdf_x1 + self.epsilon)
        H_x2 = -pdf_x2 * torch.log2(pdf_x2 + self.epsilon)
        H_x1x2 = -torch.sum(pdf_x1x2 * torch.log2(pdf_x1x2 + self.epsilon), dim=1)

        mutual_information = - (H_x1x2 - H_x1 - H_x2)

        if self.normalize:
            mutual_information = 2 * mutual_information / (H_x1 + H_x2)

        mutual_information = torch.mean(mutual_information)
        return mutual_information
