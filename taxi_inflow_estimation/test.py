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
import seaborn as sns
import argparse
from Curb_GAN import Generator

parser = argparse.ArgumentParser()
parser.add_argument("--region_i", type=int, default=10, help="i of region index.")
parser.add_argument("--region_j", type=int, default=14, help="j of region index.")
parser.add_argument("--lr", type=float, default=0.002, help="adam: learning rate")
parser.add_argument("--adj_bar", type=float, default=0.47, help="adj bar")
parser.add_argument("--init_dim", type=int, default=100, help="Dimensionality of the latent code")
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--seq_len', type=int, default=12, help='sequence length of images, which should be even nums (2,4,6,12)')
parser.add_argument('--G_input_feat', type=int, default=104, help='input features of G (include init features and num of conditions')
parser.add_argument('--G_final_feat', type=int, default=1, help='output features of G')
parser.add_argument('--G_hidden_feat', type=int, default=64, help='hidden features of G')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--num_head', type=int, default=2, help='number of heads in self-attention')
parser.add_argument('--N', type=int, default=3, help='repeating times of buiding block')


opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False

# hyper parameters
region_width = 10
region_length = 10
city_size = 50

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

if not os.path.exists('./CurbGAN_test'):
    os.mkdir('./CurbGAN_test')


# get min max of x and y
x = np.loadtxt(open('./inflow_region.csv', "rb"), delimiter=",", skiprows=0)
y = np.loadtxt(open('./label_region.csv', "rb"), delimiter=",", skiprows=0)
max_x = x.max()
max_y0 = y[:, 0].max()
max_y1 = y[:, 1].max()
max_y2 = y[:, 2].max()
max_y3 = y[:, 3].max()


################################ True Value in Reality #################################
########################################################################################
path2 = './inflow_city.csv'
inflow = np.loadtxt(open(path2, "rb"), delimiter=",", skiprows=0).reshape(-1, opt.seq_len, city_size, city_size)

path3 = './demand_city.csv'
demand = np.loadtxt(open(path3, "rb"), delimiter=",", skiprows=0).reshape(-1, opt.seq_len, city_size, city_size)

# select the test region
flow = inflow[:, :, opt.region_i:(opt.region_i + region_width), opt.region_j:(opt.region_j + region_length)]
dmd = demand[:, :, opt.region_i:(opt.region_i + region_width), opt.region_j:(opt.region_j + region_length)]

# real demand sum
dmd = np.sum(dmd, axis=2)
dmd = np.sum(dmd, axis=2)       # dmd shape: (seq_num, seq_len)

# show the statistics of the region
#################### inflow ###################
# 1. real average demand
mean_dmd = np.mean(dmd, axis=0)      # mean_dmd shape: (seq_len, )

# 2. mean inflow distribution
mean_flow = np.mean(flow, axis=0)     # mean_flow shape: (seq_len, region_length, region_width)
np.savetxt('./CurbGAN_test/real_region_mean_' + str(opt.region_i) + '_' + str(opt.region_j) + '.csv', mean_flow.reshape(-1, region_width), delimiter=',')


################################ Load Model ##########################################
########################################################################################
G = Generator(opt.G_input_feat, opt.G_hidden_feat, opt.G_final_feat, opt.num_head, opt.dropout, opt.N)  # .to(device)
G = nn.DataParallel(G).to(device)
G.load_state_dict(torch.load('./CurbGAN_train/G_params_' + str(opt.N) + '.pkl', map_location='cpu'))


########################## Prepare test data (input of G) ##########################
######################################################################################
# prepare noise
batch_x = torch.randn(flow.shape[0], opt.seq_len * opt.init_dim * region_width * region_length)
batch_x = batch_x.view(-1, opt.seq_len, region_width * region_length, opt.init_dim)
batch_x = Variable(batch_x.to(device))

# prepare adjacency matrix (need to be normalized)
all_adj = np.loadtxt(open('./all_region_inflow_adjacency.csv', "rb"), delimiter=",", skiprows=0)
all_adj = all_adj.reshape(-1, opt.seq_len, region_width * region_length, region_width * region_length)
print("all adj shape: ", all_adj.shape)
current_region_idx = int(opt.region_i * (50 - region_length + 1) + opt.region_j)
current_region_adj = all_adj[current_region_idx].reshape(-1, region_width * region_length)
print("current_region adj shape: ", current_region_adj.shape)
# normalize current_region_adj
current_region_adj[current_region_adj < opt.adj_bar] = 0   # remove the negative corr, set bar
for i in range(current_region_adj.shape[0]):
    if sum(current_region_adj[i]) == 0:
        current_region_adj[i, i - int(i / 100) * 100] = 1
    if sum(current_region_adj[i]) != 0:
        row_sum = sum(current_region_adj[i])
        current_region_adj[i] = current_region_adj[i] / row_sum
print(current_region_adj)
# repeat current_region_adj for opt.out_num times
batch_adj = np.tile(current_region_adj.reshape(opt.seq_len, region_width * region_length, region_width * region_length), (flow.shape[0], 1, 1))
batch_adj = batch_adj.reshape(-1, opt.seq_len, region_width * region_length, region_width * region_length)
batch_adj = torch.tensor(batch_adj)
batch_adj = Variable(batch_adj.to(device)).float()

# prepare conditions
region_i = opt.region_i / max_y0
region_j = opt.region_j / max_y1
test_demand = mean_dmd / max_y2     # it is a numpy of shape (seq_len,)
time = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
time = time / max_y3

y1 = torch.zeros(flow.shape[0] * opt.seq_len * region_width * region_length, 1) + region_i
y2 = torch.zeros(flow.shape[0] * opt.seq_len * region_width * region_length, 1) + region_j
batch_y = torch.cat((y1, y2), 1)

y3 = np.zeros(opt.seq_len).reshape(opt.seq_len, 1) + test_demand.reshape(opt.seq_len, 1)
y3 = np.tile(y3, (1, region_width * region_length)).reshape(opt.seq_len, region_width * region_length, 1)
y3 = np.tile(y3, (flow.shape[0], 1, 1)).reshape(flow.shape[0] * opt.seq_len * region_width * region_length, 1)
y3 = torch.from_numpy(y3).float()

batch_y = torch.cat((batch_y, y3), 1)

y4 = np.zeros(opt.seq_len).reshape(opt.seq_len, 1) + time.reshape(opt.seq_len, 1)
y4 = np.tile(y4, (1, region_width * region_length)).reshape(opt.seq_len, region_width * region_length, 1)
y4 = np.tile(y4, (flow.shape[0], 1, 1)).reshape(flow.shape[0] * opt.seq_len * region_width * region_length, 1)
y4 = torch.from_numpy(y4).float()
batch_y = torch.cat((batch_y, y4), 1).view(flow.shape[0], opt.seq_len, region_width * region_length, -1)
batch_y = Variable(batch_y.to(device)).float()


################################ Test Result ##########################################
########################################################################################
# Set model to test mode
G.eval()
test_reg = G(batch_x, batch_adj, batch_y)

##################### get generated sequences ####################
regs = test_reg.cpu().data.numpy().reshape(-1, opt.seq_len, region_width, region_length)
regs = (regs * 0.5 + 0.5) * max_x


##################### generated mean ###################
# compute the mean region of generated regions
mean_reg = np.mean(regs, axis=0)
# vmax = mean_reg.max()
np.savetxt('./CurbGAN_test/fake_region_mean_' + str(opt.region_i) + '_' + str(opt.region_j) + '_N_' + str(opt.N) + '.csv', mean_reg.reshape(-1, region_width), delimiter=',')


###################### RMSE ###############################
mean_to_mean_dist = mean_flow - mean_reg
RMSE_ls = []
for i in range(opt.seq_len):
    RMSE_ls.append(np.sqrt(np.mean((mean_to_mean_dist[i])**2)))

mean_RMSE = np.sqrt(np.mean((mean_to_mean_dist)**2))
print("RMSE in each time slot: ", RMSE_ls)
print("Average RMSE: ", mean_RMSE)


###################### MAPE ###############################
mean_to_mean_dist = np.absolute(mean_flow - mean_reg).reshape(-1, region_width)  # shape:(seq_len, region_length, region_width)
percent = mean_to_mean_dist / (mean_flow.reshape(-1, region_width))

nan_idx = np.isnan(percent)
percent[nan_idx] = 0
inf_idx = np.isinf(percent)
percent[inf_idx] = 0

detail_percent = np.sum(percent.reshape(-1, region_length, region_width), axis=1)
detail_percent = np.sum(detail_percent, axis=1)
print("MAPE in each time slot: ", detail_percent / 100)

percent = percent.sum() / 1200
print("Average MAPE: ", percent)
