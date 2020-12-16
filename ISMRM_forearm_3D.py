# np.savez('/home/oreka/Downloads/marcos_experiments_Tom2020_12_14/20201214-162403outfile-2D_Vlad.npz', data=data, kspace=kspace)
# np.savez('/home/oreka/Downloads/marcos_experiments_Tom2020_12_14/20201214-165854outfile-2D_Vlad.npz', data=data, kspace=kspace) # (30000, 32, 2)
# np.savez('/home/oreka/Downloads/marcos_experiments_Tom2020_12_16/marcos_experiments/20201215-223708outfile-3Dforearm.npz', kspace=kspace) # (30000, 32, 32) (128 res)

import numpy as np
import matplotlib.pyplot as plt
import interactive3Dplot as plt3d

def gaussianFilter(kspace, p2Param = 6):
    inputShape = np.shape(kspace)
    filterMat = 1
    for dimSize in inputShape:
        p1 = dimSize/2
        p2 = dimSize/p2Param
        filterVec = np.exp(-(np.square(np.arange(dimSize) -p1)/(p2**2)))
        filterMat = np.multiply.outer(filterMat, filterVec)
    return np.multiply(kspace, filterMat)

# workData = np.load('/home/oreka/Downloads/marcos_experiments_Tom2020_12_16/marcos_experiments/20201215-223708outfile-3Dforearm.npz') # (30000, 32)
workData = np.load('20201215-223708outfile-3Dforearm.npz') # (30000, 32)
kspace = workData['kspace'][1:-2,:]

kspaceZeros = np.dstack([np.zeros([kspace.shape[0], kspace.shape[1], 16]), kspace, np.zeros([kspace.shape[0], kspace.shape[1], 16])])
imageRecon = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(gaussianFilter(kspaceZeros[:], 2))))


def show_slice(idx):    
    plt.imshow(np.squeeze(abs(imageRecon[:,idx,:])), cmap='gray', aspect='auto')
    plt.axis('off')

if False:
    plt.figure(figsize=(12,12))

    
    for k in range(16):    
        plt.subplot(4, 4, k+1)
        show_slice(8 + k)

    plt.tight_layout()

if True:
    plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    show_slice(14)
    plt.subplot(1,2,2)
    show_slice(17)

if False:
    fig1, ax1 = plt.subplots(1,1)
    fig3D1 = plt3d.interactivePlot(fig1, ax1, np.abs(imageRecon), fov = (100,100,100))

plt.show()

# fig2, ax2 = plt.subplots(1,1)
# fig3D2 = plt3d.interactivePlot(fig2, ax2, np.log(np.abs(kspaceTrim)), fov = (100,100,100))
#
# fig3, ax3 = plt.subplots(1,1)
# fig3D3 = plt3d.interactivePlot(fig3, ax3, np.angle(kspaceTrim), fov = (100,100,100))
