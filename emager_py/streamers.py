import numpy as np
import redis
import serial
import struct

import emager_py.utils as eutils


class EmagerStreamerInterface:
    """
    Emager Streamer Interface
    """

    def read(self) -> np.ndarray:
        """
        Get a data packet from Streamer.

        Returns a numpy array of shape (n_samples, n_ch). If no samples are available, return an empty array.
        """
        pass


class RedisStreamer(EmagerStreamerInterface):
    def __init__(self, host):
        self.r = redis.Redis(host)

    def read(self):
        try:
            return np.frombuffer(
                self.r.rpop(eutils.SAMPLES_FIFO_NAME), dtype=np.int16
            ).reshape((-1, 64))
        except TypeError:
            return np.ndarray((0, 64))


class SerialStreamer(EmagerStreamerInterface):
    """
    Sensor object for data logging from HD EMG sensor
    """

    def __init__(self, serialpath, BR):
        """
        Initialize HDSensor object, open serial communication to specified port using PySerial API
        :param serialpath: (str) - Path to serial port
        :param BR: (int) - Com port baudrate
        """
        self.ser = serial.Serial(serialpath, BR, timeout=1)
        self.open()

        self.packet_size = 128
        self.ones_mask = np.ones(64, dtype=np.uint8)
        self.channel_map = (
            [10, 22, 12, 24, 13, 26, 7, 28, 1, 30, 59, 32, 53, 34, 48, 36]
            + [62, 16, 14, 21, 11, 27, 5, 33, 63, 39, 57, 45, 51, 44, 50, 40]
            + [8, 18, 15, 19, 9, 25, 3, 31, 61, 37, 55, 43, 49, 46, 52, 38]
            + [6, 20, 4, 17, 3, 23, 0, 29, 60, 35, 58, 41, 56, 47, 54, 42]
        )
        self.fmt = ">64h"

    def clear_buffer(self):
        """
        Clear the serial port input buffer.
        :return: None
        """
        self.ser.reset_input_buffer()

    def close(self):
        """
        Close serial port.
        """
        self.ser.close()

    def open(self):
        """
        Open serial port.
        """
        self.ser.open()

    def process_packet(self, data_packet, number_of_packet):
        valid_packets = []
        for i in range(number_of_packet):
            data_slice = data_packet[i * 128 : (i + 1) * 128]
            data_lsb = np.bitwise_and(data_slice[1::2], self.ones_mask)
            zero_indices = np.where(data_lsb == 0)[0]
            if len(zero_indices) == 1:
                offset = (2 * zero_indices[0] + 1) - 1
                # Second LSB bytes
                valid_packets.append(np.roll(data_slice, -offset))
        return valid_packets

    def read(self) -> np.ndarray:
        """
        Read samples from Serial device

        Returns a (n_samples, n_ch) array
        """
        bytes_available = self.ser.in_waiting
        bytes_to_read = bytes_available - (bytes_available % self.packet_size)
        samples_list = []
        if bytes_to_read > 0:
            raw_data_packet = self.ser.read(bytes_to_read)
            data_packet = np.frombuffer(raw_data_packet, dtype=np.uint8).reshape(
                (-1, self.packet_size)
            )
            number_of_packet = int(len(data_packet) / 128)
            processed_packets = self.process_packet(data_packet, number_of_packet)
            for packet in processed_packets:
                samples = np.asarray(struct.unpack(self.fmt, packet), dtype=np.int16)[
                    self.channel_map
                ]
                samples_list.append(samples)
        return np.array(samples_list)

    def __del__(self):
        self.close()


def test_acquisition():
    device = SerialStreamer("/dev/ttyACM0", 2000000)
    device.clear_buffer()
    while 1:
        sample_list = device.read()
        if len(sample_list) > 0:
            print(sample_list.shape)


if __name__ == "__main__":
    test_acquisition()
