import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import time
import numpy as np
import glob
from matplotlib import pyplot as plt
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

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import (
    DistributedSampler,
)  # Distribute data across multiple gpus
from torch.distributed import init_process_group, destroy_process_group
torch.backends.cuda.matmul.allow_tf32 = True

from.model import Generator_2, Discriminator_2, SRGAN
from loss_func import gen_loss_fn, disc_loss_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


LR_IMG_SIZE = (96,96)
HR_IMG_SIZE = (384,384)
CHANNELS_NUM = 3
BATCH_SIZE_PER_REPLICA = 32
# GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA

class CustomDataset(Dataset):
    def __init__(self, file_names, transform, target_transform):
        self.file_names = file_names
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = tv.io.read_image(self.file_names[idx])

        if self.target_transform:
            label = self.target_transform(label)

        if self.transform:
            image = self.transform(label)

        return image, label

def create_dataloader(img_filenames, transform = None, target_transform = None,
                      batch_size=12, shuffle=True, use_sampler = False, num_workers=2):
    dataset = CustomDataset(img_filenames, transform, target_transform)
    if use_sampler:
        shuffle = False
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, sampler = sampler,
                            pin_memory = True, num_workers = num_workers,
                            drop_last=True, generator=torch.Generator(device='cpu'))
    return dataloader

def train_epoch(srgan, train_dataloader, g_optimizer, d_optimizer, device):
    srgan.train()
    loss = torch.tensor([0.0] * 7, device = device)
    for idx, (lr, hr) in enumerate(train_dataloader):
        lr = lr.to(device)
        hr = hr.to(device)

        loss += srgan.train_step(lr, hr, g_optimizer, d_optimizer)

    loss = loss / len(train_dataloader)

    return loss

def test_epoch(gen, dataloader):
    gen.eval()
    with torch.no_grad():
        lr_val_samples, hr_val_samples = next(iter(dataloader))

        fig, ax = plt.subplots(3,3, figsize=(14,14))

        for i in range(3):
            ax[i,0].imshow(((lr_val_samples[i].permute(1, 2, 0)+1)*127.5).type(torch.uint8))
            ax[i,1].imshow(((gen(lr_val_samples.to(device))[i].cpu()+1)*127.5).type(torch.uint8).permute(1, 2, 0))
            ax[i,2].imshow(((hr_val_samples[i].permute(1, 2, 0)+1)*127.5).type(torch.uint8))
            ax[i,0].set_yticklabels([])
            ax[i,1].set_yticklabels([])
            ax[i,2].set_yticklabels([])
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.show()

if __name__ == '__main__':
    target_transform = tr.Compose([tr.RandomCrop(HR_IMG_SIZE),
                                tr.RandomHorizontalFlip(p=0.5),
                                tr.ToDtype(torch.float32, scale=True),
                                tr.Normalize(mean = [0.5 for _ in range(CHANNELS_NUM)],
                                                std = [0.5 for _ in range(CHANNELS_NUM)])
                                ])

    transform = tr.Compose([tr.Resize(LR_IMG_SIZE, antialias=True)])


    val_target_transform = tr.Compose([tr.Resize(HR_IMG_SIZE, antialias=True),
                                    tr.ToDtype(torch.float32, scale=True),
                                    tr.Normalize(mean = [0.5 for _ in range(CHANNELS_NUM)],
                                                    std = [0.5 for _ in range(CHANNELS_NUM)])
                                    ])

    val_transform = tr.Compose([tr.Resize(LR_IMG_SIZE, antialias=True)])

    hr_train_filenames = glob.glob('/kaggle/input/my-div2k-dataset/DIV2K_train_HR/*.png')
    hr_val_filenames = glob.glob('/kaggle/input/my-div2k-dataset/DIV2K_valid_HR/*.png')
    print(len(hr_train_filenames))
    print(len(hr_val_filenames))
    BATCH_SIZE = 16
    USE_SAMPLER = True
    train_dataloader = create_dataloader(hr_train_filenames, transform, target_transform,
                                        batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)
    val_dataloader = create_dataloader(hr_val_filenames, val_transform, val_target_transform,
                                    batch_size = 16, shuffle = False, use_sampler = USE_SAMPLER, num_workers = 4)

    from torchvision.models.vgg import vgg19
    vgg = vgg19(weights = tv.models.VGG19_Weights.DEFAULT)
    vgg = nn.Sequential(*list(vgg.features)[:31]).eval()
    for param in vgg.parameters():
        param.requires_grad = False
    vgg = vgg.to(device)

    # torch._dynamo.list_backends()
    # ['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'openxla_eval', 'tvm']
    # torch.compile(model, backend="torch_tensorrt")
    from torch._dynamo.backends.common import aot_autograd
    from functorch.compile import make_boxed_func

    def my_compiler(gm, example_inputs):
        return make_boxed_func(gm.forward)

    my_backend = aot_autograd(fw_compiler=my_compiler)  # bw_compiler=my_compiler

    generator = Generator_2()
    generator = torch.compile(generator, fullgraph=True, backend = my_backend).to(device)
    discriminator = Discriminator_2()
    discriminator = torch.compile(discriminator, fullgraph=True, backend = my_backend).to(device)
    srgan = SRGAN(generator, discriminator, gen_loss_fn, disc_loss_fn)
    srgan = torch.compile(srgan, fullgraph=True, backend = my_backend).to(device)

    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    metric_names = ['d_loss', 'g_loss', 'mse_loss', 'g_gan_loss', 'vgg_loss', 'psnr', 'ssim']

    num_epochs = 5
    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        loss = train_epoch(srgan, train_dataloader, g_optimizer, d_optimizer, device)
        print(f"ETA: {np.round(time.time() - start_time)}s {', '.join([f'{metric_name}: {value:.6f}' for metric_name, value in zip(metric_names, loss)])}")
        if (epoch +1) % 5 == 0:
            test_epoch(generator, val_dataloader)