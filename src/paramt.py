'''
params.py

Managers of all hyper-parameters

'''

import torch

epochs = 500
batch_size = 8
soft_label = False
adv_weight = 0
d_thresh = 0.7
z_dim = 256
z_dis = "norm"
model_save_step = 20
g_lr = 0.002
d_lr = 0.00001
beta = (0.25, 0.666)
cube_len = 64
leak_value = 0.2
bias = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = '../volumetric_data/'
model_dir = ''  # change it to train on other data models
output_dir = '../outputs/'
images_dir = '../test_outputs/'


def print_params():
    l = 16
    print(l * '*' + 'hyper-parameters' + l * '*')

    print('epochs =', epochs)
    print('batch_size =', batch_size)
    print('soft_labels =', soft_label)
    print('adv_weight =', adv_weight)
    print('d_thresh =', d_thresh)
    print('z_dim =', z_dim)
    print('z_dis =', z_dis)
    print('model_images_save_step =', model_save_step)
    print('data =', model_dir)
    print('device =', device)
    print('g_lr =', g_lr)
    print('d_lr =', d_lr)
    print('cube_len =', cube_len)
    print('leak_value =', leak_value)
    print('bias =', bias)

    print(l * '*' + 'hyper-parameters' + l * '*')
