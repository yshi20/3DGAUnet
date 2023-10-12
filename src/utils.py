
'''
utils.py

Some utility functions

'''

import scipy.ndimage as nd
import scipy.io as io
import matplotlib
import paramt
import tifffile as tif

if paramt.device.type != 'cpu':
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import skimage.measure as sk
from mpl_toolkits import mplot3d
import matplotlib.gridspec as gridspec
import numpy as np
from torch.utils import data
from torch.autograd import Variable
import torch
import os
import pickle
import nibabel as nib
from scipy import ndimage



def getVoxelFromMat(path, cube_len=32):
    if cube_len == 32:
        #voxels = io.loadmat(path)['instance'] # 32x32x32
        voxels = io.loadmat(path)['instance']
        #voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))

    else:
        # voxels = np.load(path) 
        voxels = io.loadmat(path)['instance'] # 64x64x64
        # voxels = np.pad(voxels, (2, 2), 'constant', constant_values=(0, 0))
        # print (voxels.shape)
        # voxels = io.loadmat(path)['instance'] # 30x30x30
        # voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
        # voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
        # print ('here')
    # print (voxels.shape)
    return voxels


def getVFByMarchingCubes(voxels, threshold=0.5):
    v, f = sk.marching_cubes_classic(voxels, level=threshold)
    return v, f


def plotVoxelVisdom(voxels, visdom, title):
    v, f = getVFByMarchingCubes(voxels)
    visdom.mesh(X=v, Y=f, opts=dict(opacity=0.5, title=title))


def SavePloat_Voxels(voxels, path, iteration):
    voxels = voxels[:8].__ge__(0.5)
    #print(len(voxels[0][0]))
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(voxels):
        #print(i,len(sample))
        x, y, z = sample.nonzero()
        #print(i, len(x))
        ax = plt.subplot(gs[i], projection='3d')
        ax.scatter(x, y, z, zdir='z', c='blue')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.set_aspect('equal')
    # print (path + '/{}.png'.format(str(iteration).zfill(3)))
    plt.savefig(path + '/{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
    #plt.imwrite(path + '/{}.tif'.format(str(iteration).zfill(3)),() photometric='minisblack' )
    plt.close()


class ShapeNetDataset(data.Dataset):

    def __init__(self, root, args, train_or_val="train"):
        
        
        self.root = root
        self.listdir = os.listdir(self.root)
        # print (self.listdir)  
        # print (len(self.listdir)) # 10668

        data_size = len(self.listdir)
#        self.listdir = self.listdir[0:int(data_size*0.7)]
        self.listdir = self.listdir[0:int(data_size)]
        
        print ('data_size =', len(self.listdir)) # train: 10668-1000=9668
        self.args = args

    def __getitem__(self, index):
        with open(self.root + self.listdir[index], "rb") as f:
            #volume = np.asarray(getVoxelFromMat(f, paramt.cube_len), dtype=np.float32)
            volume = np.asarray(getVoxelFromMat(f, paramt.cube_len), dtype=np.float32)
            # print (volume.shape)
        return torch.FloatTensor(volume)

    def __len__(self):
        return len(self.listdir)


def generateZ(args, batch):

    if paramt.z_dis == "norm":
        Z = torch.Tensor(batch, paramt.z_dim).normal_(0, 0.33).to(paramt.device)
    elif paramt.z_dis == "uni":
        Z = torch.randn(batch, paramt.z_dim).to(paramt.device).to(paramt.device)
    elif paramt.z_dis == "zero":
        Z = torch.zeros(batch, paramt.z_dim).to(paramt.device).to(paramt.device)
    else:
        print("z_dist is not normal or uniform")

    return Z
