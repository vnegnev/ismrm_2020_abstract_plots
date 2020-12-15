import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal as sig

# file0d = 'data ben Nx 128 20-12-12 16_26_56.npz'
file1d_two = 'data1d ben Nx 256 20-12-12 16_24_31.npz'
file1d_one = 'data1d ben Nx 128 20-14-12 18_54_06.npz'
# file2d = 'data2d ben Nx 166 Ny 66 TR 5 20-12-12 15_47_17.npy'

star_file = "data2d ben Nx 2000 Ny 63 TR 5.0 20-14-12 18_28_07.npy" # "data2d ben Nx 200 Ny 63 TR 5.0 20-14-12 16_50_05.npy"
tubes_file = 'data2d ben Nx 200 Ny 63 TR 5.0 20-14-12 16_50_05.npy'

def plot0d():
   data0d_dict = np.load(file0d)
   dt = data0d_dict['dt']
   nSamples = int(data0d_dict['nSamples'])
   lo_freq = data0d_dict['lo_freq']
   data = data0d_dict['data1d']

   Noise = np.abs(np.std(np.real(np.fft.fft(data))[int(data.size/2)-3:int(data.size/2)+3]))
   SNR=np.max(np.abs(np.fft.fft(data)))/Noise
   fig, (ax1, ax2, ax3) = plt.subplots(3)
   fig.suptitle('Spin Echo [n={:d}, lo_freq={:f} Mhz]\nSNR={:f}'.format(nSamples,lo_freq,SNR))
   t_axis = np.linspace(0, dt * nSamples, nSamples)  # us    
   ax1.plot(t_axis, np.abs(data)*3.3)
   ax1.set_ylabel('voltage [V]')
   ax2.set_xlabel('time [us]')
   ax2.plot(t_axis, data.real*3.3)
   ax2.set_ylabel('voltage [V]')
   f_axis = np.fft.fftshift(np.fft.fftfreq(nSamples,dt*1E-6))
   ax3.plot(f_axis,np.abs(np.fft.fftshift(np.fft.fft(data))/np.sqrt(nSamples)))
   fig.tight_layout()


def plot1d():
   data1d_dict = np.load(file1d)

   dt = data1d_dict['dt']
   nSamples = int(data1d_dict['nSamples'])
   lo_freq = data1d_dict['lo_freq']
   data1d = data1d_dict['data1d']

   fig, (ax1, ax2, ax3) = plt.subplots(3)
   fig.suptitle('Spin Echo [n={:d}, lo_freq={:f} Mhz]\n'.format(nSamples,lo_freq))
   t_axis = np.linspace(0, dt * nSamples, nSamples)  # us    
   ax1.plot(t_axis, np.abs(data1d)*3.3)
   ax1.set_ylabel('voltage [V]')
   ax2.set_xlabel('time [us]')
   ax2.plot(t_axis, data1d.real*3.3)
   ax2.set_ylabel('voltage [V]')
   f_axis = np.fft.fftshift(np.fft.fftfreq(nSamples,dt*1E-6))
   ax3.plot(f_axis,np.abs(np.fft.fftshift(np.fft.fft(data1d))/np.sqrt(nSamples)))
   fig.tight_layout()

def plot2d():
   data2d = np.load(file2d)
   plt.figure(3)
   plt.subplot(1, 3, 1)
   plt.imshow(10*np.log10(np.abs(data2d)),aspect='auto',interpolation='none')
   plt.subplot(1, 3, 2)
   plt.imshow(np.angle(data2d),aspect='auto',interpolation='none')
   plt.subplot(1, 3, 3)
   img = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(data2d))))
   plt.imshow(img, aspect='auto',cmap='gray',interpolation='none')


def plot_all():
   fig = plt.figure(figsize=(18,12))
   gs = fig.add_gridspec(4, 3)

   
   data1d_one = np.load(file1d_one)
   dt_one = data1d_one['dt']
   nSamples_one = int(data1d_one['nSamples'])
   data1d_one = data1d_one['data1d']

   ax_one = fig.add_subplot(gs[1])
   f_axis = np.fft.fftshift(np.fft.fftfreq(nSamples_one, dt_one*1e-6))
   ax_one.plot(f_axis / 1000, np.abs(np.fft.fftshift(np.fft.fft(data1d_one))/np.sqrt(nSamples_one)))
   ax_one.set_yticks([])
   # gs.tight_layout(fig)
   # plt.axis('off')
   # fig.tight_layout()

   # plt.subplots_adjust(bottom=0.2)
   
   data1d_two = np.load(file1d_two)
   dt_two = data1d_two['dt']
   nSamples_two = int(data1d_two['nSamples'])
   data1d_two = data1d_two['data1d']

   ax_two = fig.add_subplot(gs[4])
   f_axis = np.fft.fftshift(np.fft.fftfreq(nSamples_two, dt_two*1e-6))
   ax_two.plot(f_axis / 1000, np.abs(np.fft.fftshift(np.fft.fft(data1d_two))/np.sqrt(nSamples_two)))
   ax_two.set_yticks([])

   # Tubes
   # data2dOver = np.load(tubes_file)
   # import scipy.signal as sig
   # data2d = sig.decimate(data2dOver, 10, axis=1)
   data2d_tubes = np.load(tubes_file)   
   # plt.subplot(2, 3, 4)
   # plt.imshow(10*np.log10(np.abs(data2d)),aspect='auto',interpolation='none')
   # plt.axis('off')
   # plt.subplot(2, 3, 5)
   # plt.imshow(np.angle(data2d),aspect='auto',interpolation='none')
   # plt.axis('off')
   plt.subplot(2, 3, 3)
   img_tubes = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(data2d_tubes))))
   plt.imshow(img_tubes, aspect='auto',cmap='gray',interpolation='none')
   plt.axis('off')   

   # Star
   data2d_over_star = np.load(star_file)
   data2d_star = sig.decimate(data2d_over_star, 10, axis=1)
   plt.subplot(2, 3, 4)
   plt.imshow(10*np.log10(np.abs(data2d_star)),aspect='auto',interpolation='none')
   plt.axis('off')
   plt.subplot(2, 3, 5)
   plt.imshow(np.angle(data2d_star),aspect='auto',interpolation='none')
   plt.axis('off')
   plt.subplot(2, 3, 6)
   img = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(data2d_star))))
   plt.imshow(img, aspect='auto',cmap='gray',interpolation='none')
   plt.axis('off')

if __name__ == "__main__":
# plot0d()
# plot1d()
# plot2d()
   plot_all()
   # fig.tight_layout()   
   plt.show()
