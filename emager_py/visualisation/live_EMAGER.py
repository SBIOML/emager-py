import serial
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import matplotlib.colors
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from emager_py.utils.find_usb import find_psoc




def reorder(data, mask, match_result):
    '''
    Looks for mask/template matching in data array and reorders
    :param data: (numpy array) - 1D data input
    :param mask: (numpy array) - 1D mask to be matched
    :param match_result: (int) - Expected result of mask-data convolution matching
    :return: (numpy array) - Reordered data array
    '''
    data_lsb = data & np.ones(128, dtype=np.int8)
    mask_match = np.convolve(mask, np.append(data_lsb, data_lsb), 'valid')
    try:
        offset = np.where(mask_match == match_result)[0][0] - 3
    except IndexError:
        return None
    return np.roll(data, -offset)


class HDSensor(object):
    '''
    Sensor object for data logging from HD EMG sensor
    '''

    def __init__(self, serialpath, BR):
        '''
        Initialize HDSensor object, open serial communication to specified port using PySerial API
        :param serialpath: (str) - Path to serial port
        :param BR: (int) - Com port baudrate
        '''
        self.ser = serial.Serial(serialpath, BR, timeout=1)
        self.ser.close()

        self.bytes_to_read = 128
        ### ^ Number of bytes in message (i.e. channel bytes + header/tail bytes)
        self.mask = np.array([0, 2] + [0, 1] * 63)
        ### ^ Template mask for template matching on input data
        self.channelMap = [10, 22, 12, 24, 13, 26, 7, 28, 1, 30, 59, 32, 53, 34, 48, 36] + \
                          [62, 16, 14, 21, 11, 27, 5, 33, 63, 39, 57, 45, 51, 44, 50, 40] + \
                          [8, 18, 15, 19, 9, 25, 3, 31, 61, 37, 55, 43, 49, 46, 52, 38] + \
                          [6, 20, 4, 17, 2, 23, 0, 29, 60, 35, 58, 41, 56, 47, 54, 42]

        # [i for i in range(27, 32)] + [0, 1, 2] + [i for i in range(23, 27)] + \
        #                [i for i in range(3, 7)] + [22, 21, 20, 19] + [10, 9, 8, 7] + [18, 17, 16, 15, 14, 13, 12, 11]
        #                 ### ^ Channel map to hardware sensor obtained from lab tests, needed to reorder channels

    def clear_buffer(self):
        '''
        Clear the serial port input buffer.
        :return: None
        '''
        self.ser.reset_input_buffer()
        return

    def close(self):
        self.ser.close()
        return

    def open(self):
        self.ser.open()
        return

    def read(self, readtime, feedback=False, savetxt=False, savepath=None):
        '''
        Read the incoming data in com port for a given time.
        :param readtime: (int) - reading time period (seconds)
        :param feedback: (bool) - print notice upon receiving corrupted data
        :param savetxt: (bool) - save read data to csv
        :param savepath: (str) - path for saved data
        :return: (list of lists) - list of channels' listed data points (e.g. 64xN for 64 channels of N data points)
        '''
        data = [[] for i in range(64)]
        ### ^ 64 = n_channels
        self.open()
        self.clear_buffer()

        start_time = time.time()
        while (time.time() - start_time) < readtime:
            data_packet = reorder(list(self.ser.read(self.bytes_to_read)), self.mask, 63)
            if data_packet is not None:
                samples = [int.from_bytes(bytes([data_packet[i * 2], data_packet[i * 2 + 1]]), 'big', signed=True) for i
                           in range(64)]
                ### ^ Iterating over byte pairs in line, 64 => n_channels, 2 bytes per ch.

                for i, d in enumerate(data):
                    d += [samples[i]]
                    ### ^ Separating recorded data to respective channels
            elif feedback:
                print('Corrupted data. Dropped packet.')
            else:
                pass
        self.close()
        data_remap = []
        for i in self.channelMap:
            data_remap += [data[i]]
        #                 ### ^ Remapping data channels

        if savetxt:
            np.savetxt(savepath, data_remap, delimiter=',', fmt='%s')

        return data_remap  # data_remap

    def sample(self):
        '''
        Sample 1 message from com port (1 sample from each channel), retry until valid reception in case of
        corrupted data.
        :return: (list) - containing the 64 samples (1 for each channel)
        '''
        # self.open()
        self.clear_buffer()
        while (True):
            data_packet = reorder(list(self.ser.read(128)), self.mask, 63)
            if data_packet is not None:
                sample = [int.from_bytes(bytes([data_packet[i * 2], data_packet[i * 2 + 1]]), 'big', signed=True) for i
                          in
                          range(64)]  ### ^ Iterating over byte pairs in line, 32 => n_channels, 2 bytes per ch.
                # sample = [sample[i] for i in self.channelMap]
                #                             ### ^ Remapping data channels
                # self.close()
                return sample


if __name__ == '__main__':
    port = find_psoc()
    sensor = HDSensor(port, 1500000)
    print("taking baseline")
    data_baseline = np.array(sensor.read(2))
    baseline = np.median(data_baseline, axis=1)
    print(baseline.shape)
    print("taking baseline")

def f():
    data = sensor.read(0.025)  # sensor.sample()
    nb_pts = len(data[0])
    data = np.transpose(data) - np.tile(baseline, np.transpose((nb_pts,1)))
    print(np.mean(data,axis=1).shape)
    print(data)
    data = data - np.transpose(np.tile(np.mean(data,axis=1),(64,1)))
    print(data)
    print(data.shape)
    #data = np.transpose(data)
    data = np.reshape(data, (nb_pts, 4, 16))
    mean_data = np.mean(np.absolute(data), axis=0)
    print("max:", np.max(mean_data))
    print("min:", np.min(mean_data))
    # Rescale data and makes colors
    # min = 100  # -32769
    # max = 22000  # 32769
    min = 10  # -32769
    max = 700  # 32769
    rescaled_data = (mean_data - min) / (max - min)
    colors = cm.viridis(rescaled_data)
    # colors = cm.viridis(mean_data)
    return colors

fig = plt.figure()
im2 = plt.imshow(f(), animated=True, cmap=cm.viridis) #plt.scatter

#1. periode temps tous les gestes (non supervisé)
#2. PCA avec les données
#3. model temps réels PCA


def updatefig(*args):
    im2.set_array(f())
    return im2,


ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
plt.title('Muscle activation')
plt.show()
