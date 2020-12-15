import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import csv
import pdb
st = pdb.set_trace

def scope_plot():
    data = np.genfromtxt("RefCurve_2020-12-14_0_175732.Wfm.csv", delimiter=',')
    dt = 3.333e-7 * 1e3 # msec
    t = np.arange(data.shape[0])*dt

    # filter scope data
    if False:
        data_filt = np.empty_like(data)
    
        for k, d in enumerate(data.T):
            data_filt[:,k] = sig.convolve(d, np.ones(10)/10, 'same')

        data = data_filt

    fig, axs = plt.subplots(4, 1, figsize=(18,8), sharex=True)
    data_mapping = [0, 1, 3, 2] # which data gets plotted on which axis -- want the RX to appear last
    for k, ax in enumerate(axs):
        ax.plot(t, data[:,data_mapping[k]])
        ax.set_xlim([0, 20])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    scope_plot()
