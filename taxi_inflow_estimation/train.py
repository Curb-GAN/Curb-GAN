import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torchvision import datasets
import matplotlib.pyplot as plt
import imageio
import itertools
import numpy as np
import struct
import argparse
from Curb_GAN import Generator
from Curb_GAN import Discriminator
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=1500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.002, help="adam: learning rate")
parser.add_argument("--beta1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--beta2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--adj_bar", type=float, default=0.47, help="adj bar")
parser.add_argument("--init_dim", type=int, default=100, help="dimensionality of the latent code")
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--seq_len', type=int, default=12, help='sequence length of images, which should be even nums (2,4,6,12)')

parser.add_argument('--num_head', type=int, default=2, help='number of heads in self-attention')

parser.add_argument('--D_input_feat', type=int, default=5, help='input features of D (include init features and num of conditions')
parser.add_argument('--D_final_feat', type=int, default=1, help='output features of D')
parser.add_argument('--D_hidden_feat', type=int, default=16, help='hidden features of D')


parser.add_argument('--G_input_feat', type=int, default=104, help='input features of G (include init features and num of conditions')
parser.add_argument('--G_final_feat', type=int, default=1, help='output features of G')
parser.add_argument('--G_hidden_feat', type=int, default=64, help='hidden features of G')
parser.add_argument('--N', type=int, default=3, help='repeating times of buiding block')

#parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--l2', type=float, default=0.1, help='l2 penalty')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
#parser.add_argument('--clip_value', type=float, default=20, help='gradient clipping value')

opt = parser.parse_args()
print(opt)
print(opt.adj_bar)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False

region_width = 10
region_length = 10

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

if not os.path.exists('./CurbGAN_train'):
    os.mkdir('./CurbGAN_train')


################################ Data cleaning ##########################################
########################################################################################
x = np.loadtxt(open('./inflow_region.csv', "rb"), delimiter=",", skiprows=0)
y = np.loadtxt(open('./label_region.csv', "rb"), delimiter=",", skiprows=0)
adj = np.loadtxt(open('./inflow_region_adjacency_' + str(opt.adj_bar) + '.csv', "rb"), delimiter=",", skiprows=0)

x = x.reshape(-1, region_width * region_length, 1)
adj = adj.reshape(-1, region_width * region_length, region_width * region_length)


# remove the last 10 rows and related regions due to bad data quality
for i in [35, 40]:
    for j in range(0, 41, 5):
        arr1 = np.where(y[:, 0] == i)[0]
        arr2 = np.where(y[:, 1] == j)[0]
        idx = np.intersect1d(arr1, arr2)

        x = np.delete(x, idx, axis=0)
        y = np.delete(y, idx, axis=0)
        adj = np.delete(adj, idx, axis=0)


x = x.reshape(-1, opt.seq_len, region_width * region_length, 1)
adj = adj.reshape(-1, opt.seq_len, region_width * region_length, region_width * region_length)


################################ Data loading ##########################################
########################################################################################
# min-max normalization
def min_max_normal(x):
    max_x = x.max()
    min_x = x.min()
    x = (x - min_x) / (max_x - min_x)
    return x


# input normalization
max_num = x.max()
x = x / x.max()
x = (x - 0.5) / 0.5
x = torch.tensor(x)

adj = torch.tensor(adj)

y = y.repeat(region_width * region_length, axis=0)

# label normalization
for i in [0, 1, 2, 3]:
    y[:, i] = min_max_normal(y[:, i])

y = y.reshape(-1, opt.seq_len, region_width * region_length, 4)
y = torch.tensor(y)

dataset = Data.TensorDataset(x, y, adj)
train_loader = Data.DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True)


################################ Temporal GAN ##########################################
########################################################################################
D = Discriminator(opt.D_input_feat, opt.D_hidden_feat, opt.D_final_feat, opt.num_head, opt.dropout, opt.N)
G = Generator(opt.G_input_feat, opt.G_hidden_feat, opt.G_final_feat, opt.num_head, opt.dropout, opt.N)

if torch.cuda.device_count() > 1:
    print("number of GPU: ", torch.cuda.device_count())
    D = nn.DataParallel(D).to(device)
    G = nn.DataParallel(G).to(device)
if torch.cuda.device_count() == 1:
    D = D.to(device)
    G = G.to(device)

opt_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.l2)
opt_G = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.l2)

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

################################ Loss histgram ##########################################
########################################################################################


def show_train_hist(hist, path='Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(path)


def to_img(x, max_num):
    x = x.cpu().data.numpy().reshape(-1, opt.seq_len, region_width, region_length)
    x = (x * 0.5 + 0.5) * max_num
    x = x[0, :, :, :]
    return x


################################ Training ##########################################
########################################################################################
train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []

# training
for epoch in range(opt.epoch):
    D_losses = []
    G_losses = []

    # learning rate decay
    if epoch == 10:  # or epoch == 15:
        opt_G.param_groups[0]['lr'] /= 10
        opt_D.param_groups[0]['lr'] /= 10

    for step, (b_x, b_y, b_adj) in enumerate(train_loader):
        ######################### Train Discriminator #######################
        D.zero_grad()
        num_seq = b_x.size(0)              # batch size of sequences

        real_seq = Variable(b_x.to(device)).float()     # put tensor in Variable
        seq_label = Variable(b_y.to(device)).float()
        seq_adj = Variable(b_adj.to(device)).float()
        prob_real_seq_right_pair = D(real_seq, seq_adj, seq_label)

        noise = torch.randn(num_seq, opt.seq_len * opt.init_dim * region_length * region_width).view(num_seq, opt.seq_len, region_length * region_width, opt.init_dim)
        noise = Variable(noise.to(device))  # randomly generate noise

        fake_seq = G(noise, seq_adj, seq_label)
        prob_fake_seq_pair = D(fake_seq, seq_adj, seq_label)

        # sample real seqs from database(just shuffle this batch seqs)
        shuffled_row_idx = torch.randperm(num_seq)
        real_shuffled_seq = b_x[shuffled_row_idx]
        real_shuffled_seq = Variable(real_shuffled_seq.to(device)).float()
        shuffled_adj = b_adj[shuffled_row_idx]
        shuffled_adj = Variable(shuffled_adj.to(device)).float()

        prob_real_seq_wrong_pair = D(real_shuffled_seq, shuffled_adj, seq_label)

        D_loss = - torch.mean(torch.log(prob_real_seq_right_pair) +
                              torch.log(1. - prob_fake_seq_pair) + torch.log(1. - prob_real_seq_wrong_pair))

        D_loss.backward()
        opt_D.step()

        D_losses.append(D_loss.item())

        ########################### Train Generator #############################
        G.zero_grad()
        noise2 = torch.randn(num_seq, opt.seq_len * opt.init_dim * region_length * region_width).view(num_seq, opt.seq_len, region_length * region_width, opt.init_dim)
        noise2 = Variable(noise2.to(device))  # randomly generate noise

        # create random label
        y_real = Variable(torch.ones(num_seq).to(device))
        G_result = G(noise2, seq_adj, seq_label)
        D_result = D(G_result, seq_adj, seq_label).squeeze()

        G_loss = BCE_loss(D_result, y_real)

        G_loss.backward()
        opt_G.step()

        G_losses.append(G_loss.item())

    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))

    if (epoch + 1) % 10 == 0:
        fake_seqs = to_img(fake_seq, max_num)
        sns.heatmap(fake_seqs.reshape(-1, region_width))
        plt.savefig('./CurbGAN_train/fake_seqs-{}.png'.format(epoch + 1))
        plt.close()

show_train_hist(train_hist, path='./CurbGAN_train/train_loss_hist_' + str(opt.N) + '.png')
torch.save(G.state_dict(), './CurbGAN_train/G_params_' + str(opt.N) + '.pkl')   # save parameters
torch.save(D.state_dict(), './CurbGAN_train/D_params_' + str(opt.N) + '.pkl')
