import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# import ignite
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
# import torch_tensorrt
import torchvision as tv
import torchvision.transforms.v2 as tr
from torchmetrics.image import PeakSignalNoiseRatio as PSNR, StructuralSimilarityIndexMeasure as SSIM
from torchsummary import summary

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import (
    DistributedSampler,
)  # Distribute data across multiple gpus
from torch.distributed import init_process_group, destroy_process_group
torch.backends.cuda.matmul.allow_tf32 = True

# torch.multiprocessing.set_start_method('spawn', force=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.set_default_device(device)
print(device)

class Generator_2(nn.Module):
    def __init__(self, num_blocks=16):
        super(Generator_2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

        self.residual_blocks = nn.Sequential(*[ResidualBlock() for _ in range(num_blocks)])

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.subpixel1 = nn.PixelShuffle(upscale_factor=2)
        self.relu1 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.subpixel2 = nn.PixelShuffle(upscale_factor=2)
        self.relu2 = nn.ReLU()

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        temp = x

        x = self.residual_blocks(x)
        x = self.bn1(self.conv2(x))
        x = x + temp

        x = self.relu1(self.subpixel1(self.conv3(x)))
        x = self.relu2(self.subpixel2(self.conv4(x)))

        x = torch.tanh(self.conv5(x))
        return x

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        z = self.relu(self.bn1(self.conv1(x)))
        z = self.bn2(self.conv2(z))
        x = x + z
        return x

class Discriminator_2(nn.Module):
    def __init__(self):
        super(Discriminator_2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.leaky1 = nn.LeakyReLU(negative_slope=0.2)

        self.convblock = nn.Sequential(
            ConvolutionBlock(64,64, 3, 2),
            ConvolutionBlock(64, 128, 3, 2),
            ConvolutionBlock(128, 128, 3, 2),
            ConvolutionBlock(128, 256, 3, 2),
            ConvolutionBlock(256, 256, 3, 2),
            ConvolutionBlock(256, 512, 3, 1),
            ConvolutionBlock(512, 512, 3, 1)
        )

        self.flatten = nn.Flatten()
        self.Dense1 = nn.Linear(512 * 12 * 12, 256)
        self.leaky2 = nn.LeakyReLU(negative_slope=0.2)
        self.Dense2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky1(self.conv1(x))
        x = self.convblock(x)
        x = self.flatten(x)
        x = self.leaky2(self.Dense1(x))
        logits = self.Dense2(x)
        n = self.sigmoid(logits)
        return n, logits

class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides):
        super(ConvolutionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=strides, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.leaky_relu(self.batch_norm(self.conv(x)))
        return x

class SRGAN(nn.Module):
    def __init__(self, generator, discriminator, generator_loss, discriminator_loss):
        super(SRGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.loss_fn1 = nn.BCEWithLogitsLoss()
        self.loss_fn2 = nn.MSELoss()
        self.psnr_metric = PSNR(data_range = 2)
        self.ssim_metric = SSIM(data_range = 2)

    def forward(self, x):
        return self.generator(x)

    def train_step(self, lr, hr, g_optimizer, d_optimizer):
        self.generator.train()
        self.discriminator.train()

        # Generator
        fake_hr = self.generator(lr)
        mse_loss, g_gan_loss, vgg_loss = self.generator_loss(self.discriminator, self.loss_fn1, self.loss_fn2, hr, fake_hr)
        g_loss = mse_loss + g_gan_loss + vgg_loss

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # Discriminator
        d_hr_pred, d_hr_pred_logits = self.discriminator(hr)
        d_hr_fake_pred, d_hr_fake_pred_logits = self.discriminator(fake_hr.detach())  # Detach to prevent gradients from flowing back

        d_loss = self.discriminator_loss(self.loss_fn1, d_hr_pred_logits, d_hr_fake_pred_logits)

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # PSNR and SSIM will act as a loss function and will contain the gradient graphs that can make some error as OOM
        # so we need detach fake_hr or use no_grad to ignore gradient
        with torch.no_grad():
            psnr = self.psnr_metric(hr, fake_hr)
            ssim = self.ssim_metric(hr, fake_hr)

        return torch.stack([g_loss, d_loss, mse_loss, g_gan_loss, vgg_loss, psnr, ssim])


