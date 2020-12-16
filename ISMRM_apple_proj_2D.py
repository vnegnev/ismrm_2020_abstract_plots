# np.savez('/home/oreka/Downloads/marcos_experiments_Tom2020_12_14/20201214-162403outfile-2D_Vlad.npz', data=data, kspace=kspace)
import numpy as np
import matplotlib.pyplot as plt

workData = np.load('20201214-162403outfile-2D_Vlad.npz') # (30000, 32)
kspace = workData['kspace'][2:-1,:]
plt.figure(figsize=(12,12))
plt.subplot(2, 2, 1)
Y = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(kspace)))
img = np.abs(Y)
plt.imshow(np.log10(np.abs(kspace))) #, cmap='gray')
plt.axis('off')
# plt.title('k-Space Magn.')
plt.subplot(2, 2, 2)
kSpaceAngl = np.angle(kspace) # np.angle(Y)
plt.imshow(kSpaceAngl) #, cmap='gray')
plt.axis('off')
# plt.title('k-Space phase')
plt.subplot(2, 2, 3)
plt.imshow(img, cmap='gray')
plt.axis('off')
# plt.title('Image')

plt.subplot(2, 2, 4)
imgAngl = np.angle(Y*np.exp(1j*np.pi)) # np.angle(Y)
plt.imshow(imgAngl, cmap='gray')
plt.axis('off')
# plt.title('Image phase')
plt.show()
