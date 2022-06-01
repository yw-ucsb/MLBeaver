import matplotlib.pyplot as plt
import torch
import numpy as np
import time
import os
import dataparser
import itertools

from torch import optim
from torch.nn import functional as F
import transforms, distributions, flows
from utils import torchutils
from torch.utils import data
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

np.set_printoptions(precision=4, suppress=True)
torch.set_printoptions(precision=4, sci_mode=False)


np.random.seed(1137)
torch.manual_seed(114514)

# Setup device;
assert torch.cuda.is_available()
device = torch.device('cuda')
torch.set_default_tensor_type(torch.FloatTensor)
torch.cuda.empty_cache()

dataset_name_option = ['power', 'gas', 'hepmass', 'miniboone', 'bsds300']

train_option = [dataset_name_option]

os.environ['DATAROOT'] = ''

summary_path = ''

# Define batch size;
batch_size = 256
# Define validation batch size;
val_batch_size = 1024

# Flow configuration;
num_transformation = 10
hidden_feature = 256
block_feature = 2
num_knot = 8

grad_norm_clip_value = 5.0

# Training configuration;
num_iter = 50000
val_interval = 250
lr = 0.0005

with open(summary_path, 'a') as fp:
    fp.write('Experiment run on Gauss\n')
    fp.write('Number of iterations: {}, learning rate: {}\n'.format(num_iter, lr))

for train_setting in itertools.product(*train_option):
    print('Current setting is:', train_setting)
    # Construct data;
    data_train = dataparser.load_dataset(train_setting[0], split='train')
    train_loader = data.DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True)
    train_generator = dataparser.batch_generator(train_loader)
    print('num_train:', len(data_train))

    data_val = dataparser.load_dataset(train_setting[0], split='val')
    val_loader = data.DataLoader(data_val, batch_size=val_batch_size, shuffle=True, drop_last=True)
    val_generator = dataparser.batch_generator(val_loader)
    print('num_val:', len(data_val))

    data_test = dataparser.load_dataset(train_setting[0], split='test')
    test_loader = data.DataLoader(data_test, batch_size=val_batch_size, shuffle=True, drop_last=True)
    print('num_test:', len(data_test))

    # Extract the dimension of the dataset as feature;
    feature = data_train.dim
    print('Dimension of dataset:', feature)

    model_path = '{}_best_model.t'.format(train_setting[0])

    # Flow configuration;
    # Base distribution;
    base_dist = distributions.StandardNormal(shape=[feature])

    # Constructing transformation;
    transform = []

    # Composition of the final transformation;
    Transform = transforms.CompositeTransform(transform)

    flow = flows.Flow(Transform, base_dist).to(device)
    n_params = torchutils.get_num_parameters(flow)
    print('There are {} trainable parameters in this model.'.format(n_params))

    optimizer = optim.Adam(flow.parameters(), lr=lr)

    tbar = tqdm(range(num_iter))
    train_loss = np.zeros(shape=(num_iter))
    val_score = np.zeros(shape=(int(num_iter / val_interval)))

    start = time.time()

    t_val = 0
    count_val = 0
    best_val_score = 1000

    for i in tbar:
        # Training iterations;
        flow.train()
        batch = next(train_generator).to(device)
        # print(batch.shape)
        optimizer.zero_grad()
        loss = -flow.log_prob(inputs=batch).mean()
        # print('Current loss:', loss.detach().cpu().numpy())
        # train_loss[i] = loss.detach().numpy()
        loss.backward()
        clip_grad_norm_(flow.parameters(), grad_norm_clip_value)
        optimizer.step()

        # Validation;
        if (i + 1) % val_interval == 0:
            flow.eval()
            val_start = time.time()
            with torch.no_grad():
                # compute validation score
                running_val_log_density = 0
                for val_batch in val_loader:
                    log_density_val = -flow.log_prob(val_batch.to(device).detach())
                    mean_log_density_val = torch.mean(log_density_val).detach()
                    running_val_log_density += mean_log_density_val
                running_val_log_density /= len(val_loader)
            if running_val_log_density < best_val_score:
                best_val_score = running_val_log_density
                torch.save(flow.state_dict(), model_path)
            val_score[count_val] = running_val_log_density.cpu().detach().numpy()
            print('Current validation score is:', val_score[count_val])
            val_end = time.time()
            t_val = t_val + (val_end - val_start)
            count_val += 1

    end = time.time()
    elapsed_time = end - start
    training_time = elapsed_time - t_val
    t_val = t_val / count_val

    print('Total time:', elapsed_time)
    print('Training time:', training_time)
    print('Validation time:', t_val)

    # Test based on best model;
    flow.load_state_dict(torch.load(model_path))
    flow.eval()

    # calculate log-likelihood on test set
    with torch.no_grad():
        log_likelihood = torch.Tensor([]).to(device)
        for batch in tqdm(test_loader):
            log_density = flow.log_prob(batch.to(device))
            log_likelihood = torch.cat([
                log_likelihood,
                log_density
            ])
    mean_log_likelihood = -log_likelihood.mean()
    std_log_likelihood = log_likelihood.std()

    with open(summary_path, 'a') as fp:
        fp.write('Current dataset is {}\n'.format(train_setting[0]))
        fp.write(
            'Number of layers: {}, hidden feature: {}, block feature: {}\n'.format(num_transformation, hidden_feature,
                                                                                   block_feature))
        fp.write('Best validation score {:.2f}\n'.format(best_val_score))
        fp.write('Test performance is {:.2f}+-{:.2f}\n'.format(mean_log_likelihood,
                                                               2 * std_log_likelihood / np.sqrt(len(data_test))))
        fp.write('Total_time: {:.2f}\n'.format(elapsed_time))
        fp.write('Training time: {:.2f}\n'.format(training_time))
        fp.write('Validation time: {:.2f}\n'.format(t_val))

# plt.plot(train_loss)
# plt.show()
