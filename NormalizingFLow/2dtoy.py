from matplotlib import cm, pyplot as plt
import torch
import numpy as np
import time
import os

import nn.nets as net
import dataparser
import itertools

from torch import optim
import transforms
import distributions
import flows
from utils import torchutils
from torch.utils import data
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

np.set_printoptions(suppress=True)


np.random.seed(1137)
torch.manual_seed(114514)

# Setup device;
assert torch.cuda.is_available()
device = torch.device('cuda')
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# dataset_name = 'gaussian'
dataset_name = 'checkerboard'


os.environ['DATAROOT'] = ''


summary_path = ''

# Define batch size;
batch_size = 512
n_data_points = int(1e6)
num_transformation = 4

feature = 2

lr = 0.001
num_iter = 10000
val_interval = 250

# create data
train_dataset = dataparser.load_plane_dataset(dataset_name, n_data_points)
train_loader = dataparser.InfiniteLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_epochs=None, generator=torch.Generator(device='cuda')
)

# Generate test grid data
num_points_per_axis = 512
limit = 4
bounds = np.array([
    [-limit, limit],
    [-limit, limit]
])
grid_dataset = dataparser.TestGridDataset(
    num_points_per_axis=num_points_per_axis,
    bounds=bounds
)
grid_loader = data.DataLoader(
    dataset=grid_dataset,
    batch_size=1000,
    drop_last=False, generator=torch.Generator(device='cuda')
)

# create model
base_dist = distributions.StandardNormal(shape=[feature])

# Constructing transformation;
transform = []

# Composition of the final transformation;
Transform = transforms.CompositeTransform(transform)

flow = flows.Flow(Transform, base_dist).to(device)

optimizer = optim.Adam(flow.parameters(), lr=lr)

n_params = torchutils.get_num_parameters(flow)
print('There are {} trainable parameters in this model.'.format(n_params))


tbar = tqdm(range(num_iter))
for step in tbar:
    flow.train()
    optimizer.zero_grad()

    batch = next(train_loader).to(device)
    log_density = flow.log_prob(batch)
    loss = - torch.mean(log_density)
    loss.backward()
    optimizer.step()

    if (step + 1) % val_interval == 0:
        flow.eval()
        print('Current loss is:', loss.detach())
        # print('Current W:', transform[0]._transforms[1].weight())
        log_density_np = []
        for batch in grid_loader:
            batch = batch.to(device)
            log_density = flow.log_prob(batch)
            log_density_np = np.concatenate(
                (log_density_np, torchutils.tensor2numpy(log_density))
            )

        figure, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
        figure.suptitle('Dataset is {}, loss is {:.3f}'.format(dataset_name, loss.cpu().detach().numpy()))

        cmap = cm.magma
        axes[0].hist2d(torchutils.tensor2numpy(train_dataset.data[:, 0]),
                       torchutils.tensor2numpy(train_dataset.data[:, 1]),
                       range=bounds, bins=512, cmap=cmap, rasterized=False)
        axes[0].set_xlim(bounds[0])
        axes[0].set_ylim(bounds[1])
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        axes[1].pcolormesh(grid_dataset.X, grid_dataset.Y,
                           np.exp(log_density_np).reshape(grid_dataset.X.shape),
                           cmap=cmap)
        axes[1].set_xlim(bounds[0])
        axes[1].set_ylim(bounds[1])
        axes[1].set_xticks([])
        axes[1].set_yticks([])


        plt.tight_layout()

        path = summary_path + '{}'.format(dataset_name)
        plt.savefig(path, dpi=600)
        plt.close()







