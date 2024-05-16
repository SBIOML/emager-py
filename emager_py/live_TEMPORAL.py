import serial
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import collections
from matplotlib.colors import LogNorm
import matplotlib.colors
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import signal
from scipy.fft import fft, ifft, fftfreq



def reorder(data, mask, match_result):
    '''
    Looks for mask/template matching in data array and reorders
    :param data: (numpy array) - 1D data input
    :param mask: (numpy array) - 1D mask to be matched
    :param match_result: (int) - Expected result of mask-data convolution matching
    :return: (numpy array) - Reordered data array
    '''
    number_of_packet = int(len(data)/128)
    roll_data = []
    for i in range(number_of_packet):
        data_lsb = data[i*128:(i+1)*128] & np.ones(128, dtype=np.int8)
        mask_match = np.convolve(mask, np.append(data_lsb, data_lsb), 'valid')
        try:
            offset = np.where(mask_match == match_result)[0][0] - 3
        except IndexError:
            return None
        roll_data.append(np.roll(data[i*128:(i+1)*128], -offset))
    return roll_data


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


    def live_read(self, feedback=False, savetxt=False, savepath=None, firstTime=False):
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
        if firstTime:
            self.open()
            self.clear_buffer()
            time.sleep(0.0005)

        data_packet = None
        while data_packet is None:
            bytes_available = self.ser.inWaiting()
            bytesToRead = bytes_available - (bytes_available % 128)
            data_packet = reorder(list(self.ser.read(bytesToRead)), self.mask, 63)
        for packet in data_packet:
            samples = [int.from_bytes(bytes([packet[i * 2], packet[i * 2 + 1]]), 'big', signed=True) for i in range(64)]
            for i, d in enumerate(data):
                d += [samples[i]]
        ### ^ Iterating over byte pairs in line, 64 => n_channels, 2 bytes per ch.
            ### ^ Separating recorded data to respective channels
        data_remap = []
        for i in self.channelMap:
            data_remap += [data[i]]
        #                 ### ^ Remapping data channels

        return data_remap  # data_remap


if __name__ == '__main__':
    sensor = HDSensor('COM13', 1500000)


def preprocess_data(new_data, nb_pts):
    # add to buffer
    data_buffer.extend(new_data.tolist())
    #print(data_buffer.__len__())
    # remove DC

    # filter 60 Hz
    # Rectification
    # MAV
firstGo = True
channels_of_interest = [2,9] # give 2
#channels_of_interest = [3, 38] # give 2
data_buffer = collections.deque(maxlen=1024)
extension = np.zeros((3,64)).tolist()
data_buffer.extend(extension)
#print(data_buffer)

def data_gen():
    global firstGo
    data = sensor.live_read(firstTime=firstGo)  # sensor.sample()
    firstGo = False
    nb_pts = len(data[0])
    data = np.transpose(data)
    mod_data = preprocess_data(data, nb_pts)
    y1 = 0.000195 * data[:, channels_of_interest[0]]
    y2 = 0.000195 * data[:, channels_of_interest[1]]
    t = np.linspace(0, nb_pts/1000, nb_pts) + data_gen.time
    data_gen.time = t[-1]
    return t, y1, y2

data_gen.time = 0

# create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2,1)

# intialize two line objects (one in each axes)
line1, = ax1.plot([], [], lw=2, label='Flexor')
line2, = ax2.plot([], [], lw=2, color='r', label='extensor')
line = [line1, line2]

# the same axes initalizations as before (just now we do it for both of them)
for ax in [ax1, ax2]:
    ax.set_ylim(-2, 2)
    #ax.set_xlim(0, 5)
    ax.grid()

# initialize the data arrays
xdata, y1data, y2data = [], [], []
def run(i):
    # update the data
    t, y1, y2 = data_gen()
    xdata.extend(t)
    y1data.extend(y1)
    y2data.extend(y2)

    # axis limits checking. Same as before, just for both axes
    if len(xdata) > 2000:
        xmin = xdata[-2000]
        xmax = xdata[-1]
    else:
        xmin = 0
        xmax = xdata[-1] + 1
    for ax in [ax1, ax2]:
        ax.set_xlim(xmin, xmax)
        ax.figure.canvas.draw()

    # update the data of both line objects
    line[0].set_data(xdata, y1data)
    line[1].set_data(xdata, y2data)

    return line

ani = animation.FuncAnimation(fig, run, blit=True, interval=10, repeat=False)
plt.show()


# # Parameters
# multiple = True # allows 2 channel display (example: flexor and extensor)
# differential = False
# channel_of_interest = 45
# channels_of_interest = [19, 40] # give 2
#
# firstGo = True
# time_ms = [0]
#
# data_buffer0 = [0]
# data_buffer1 = [0]
#
# # create a figure with two subplots
# fig, (ax1, ax2) = plt.subplots(2,1)
#
# # intialize two line objects (one in each axes)
# # line1, = ax1.plot([], [], lw=2)
# # line2, = ax2.plot([], [], lw=2, color='r')
# # line = [line1, line2]
#
# # the same axes initalizations as before (just now we do it for both of them)
# # for ax in [ax1, ax2]:
# #     ax.set_ylim(-6, 6)
# #     #ax.set_xlim(0, 5)
# #     ax.grid()
#
# def f(i):
#     global firstGo
#     global data_buffer0
#     global data_buffer1
#     global time_ms
#     data = sensor.live_read(firstTime=firstGo)  # sensor.sample()
#     firstGo = False
#     nb_pts = len(data[0])
#     data = np.transpose(data)
#     x = np.linspace(1, nb_pts, nb_pts) + time_ms[-1]
#     y0 = 0.000195 * data[:, channels_of_interest[0]]
#     y1 = 0.000195 * data[:, channels_of_interest[1]]
#     time_ms.extend(x)
#     data_buffer0.extend(y0)
#     data_buffer1.extend(y1)
#     if len(time_ms) > 300:
#         time_ms = time_ms[-300:]
#         data_buffer0 = data_buffer0[-300:]
#         data_buffer1 = data_buffer1[-300:]
#     plt.cla()
#     ax1.plot(time_ms, data_buffer0, label='Live EMG Channel' + str(channels_of_interest[0]))
#     ax1.set_xlim(-2, 2)
#     ax1.set_xlabel('time (ms)')
#     ax1.set_ylabel('voltage (mV)')
#     ax2.plot(time_ms, data_buffer1, label='Live EMG Channel' + str(channels_of_interest[1]))
#     ax2.set_xlim(-2, 2)
#     ax2.set_xlabel('time (ms)')
#     ax2.set_ylabel('voltage (mV)')
#     plt.title("Live EMG" + str(channels_of_interest[0]) + ", " + str(channels_of_interest[1]))
#     #plt.grid()
#     #plt.tight_layout()
#
# ani = animation.FuncAnimation(plt.gcf(), f, interval=100)
#
# plt.tight_layout()
# plt.show()




