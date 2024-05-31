import numpy as np
import serial
import struct
import socket
import subprocess as sp
import time
import sys
import logging as log
from typing import Union

import emager_py.utils.emager_redis as er


class EmagerStreamerInterface:
    """
    Emager Streamer Interface
    """

    def read(self) -> np.ndarray:
        """
        Get a data packet from Streamer.

        Returns a numpy array of shape (n_samples, n_ch). If no samples are available, return an empty array.
        """
        raise NotImplementedError(
            "read() method must be implemented for the StreamerInterface."
        )

    def write(self, data: np.ndarray, labels: Union[np.ndarray, None] = None):
        """
        Write data to the Stream.
        """
        pass

    def configure(self, **kwargs):
        """
        Configure the Streamer
        """
        pass

    def clear(self):
        """
        Clear the Streamer's data
        """
        pass

    def set_len(self, len: int):
        """
        Set the number of samples to generate
        """
        pass

    def __len__(self):
        pass


class RedisStreamer(EmagerStreamerInterface):
    def __init__(self, host: str, labelling: bool = False):
        self.r = er.EmagerRedis(host)
        self.set_labelling(labelling)

    def read(self):
        data = self.r.pop_sample(self.labelling)
        if data == ():
            return np.ndarray((0, 64))

        if self.labelling:
            return data
        else:
            return data[0]

    def write(self, data: np.ndarray, labels: Union[np.ndarray, None] = None):
        if self.labelling:
            self.r.push_sample(data, labels)
        else:
            self.r.push_fifo(self.r.SAMPLES_FIFO_KEY, data.astype(np.int16).tobytes())

    def set_labelling(self, labelling):
        self.labelling = labelling

    def set_len(self, len):
        self.r.set(self.r.GENERATED_SAMPLES_KEY, len)

    def __len__(self):
        return self.r.get_int(self.r.GENERATED_SAMPLES_KEY)

    def configure(self, **kwargs):
        for key, value in kwargs.items():
            self.r.set(key, value)

    def clear(self):
        self.r.clear_data()


class SerialStreamer(EmagerStreamerInterface):
    """
    Sensor object for data logging from HD EMG sensor
    """

    def __init__(self, port, baud=115200, virtual=False):
        """
        Initialize HDSensor object, open serial communication to specified port using PySerial API
        :param port: (str) - Path to serial port
        :param baud: (int) - Com port baudrate
        :param virtual: (bool) - If True, virtual serial port
        """
        if virtual:
            log.info("Sleeping 2s to give time for virtual serial port to start")
            time.sleep(2)

        self.virtual = virtual
        self.ser = serial.Serial(port, baud, timeout=1)
        self.packet_size = 128
        self.ones_mask = np.ones(64, dtype=np.uint8)
        self.channel_map = (
            [10, 22, 12, 24, 13, 26, 7, 28, 1, 30, 59, 32, 53, 34, 48, 36]
            + [62, 16, 14, 21, 11, 27, 5, 33, 63, 39, 57, 45, 51, 44, 50, 40]
            + [8, 18, 15, 19, 9, 25, 3, 31, 61, 37, 55, 43, 49, 46, 52, 38]
            + [6, 20, 4, 17, 3, 23, 0, 29, 60, 35, 58, 41, 56, 47, 54, 42]
        )
        self.fmt = ">64h"

    def close(self):
        """
        Close serial port.
        """
        if self.ser:
            if self.ser.is_open:
                self.ser.close()

    def open(self):
        """
        Open serial port.
        """
        if self.ser:
            if not self.ser.is_open:
                self.ser.open()

    def process_packet(self, data_packet, number_of_packet, validate=True):
        valid_packets = []
        for i in range(number_of_packet):
            data_slice = data_packet[i * 128 : (i + 1) * 128]
            data_lsb = np.bitwise_and(data_slice[1::2], self.ones_mask)
            zero_indices = np.where(data_lsb == 0)[0]
            if len(zero_indices) == 1 and validate:
                offset = (2 * zero_indices[0] + 1) - 1
                # Second LSB bytes
                valid_packets.append(np.roll(data_slice, -offset))
        return valid_packets

    def read(self) -> np.ndarray:
        """
        Read samples from Serial device

        Returns a (n_samples, n_ch) array
        """
        
        self.open()

        bytes_available = self.ser.in_waiting
        bytes_to_read = bytes_available - (bytes_available % self.packet_size)
        # Wait to have a complete data packet
        while bytes_to_read < self.packet_size:
            bytes_available = self.ser.in_waiting
            bytes_to_read = bytes_available - (bytes_available % self.packet_size)
            time.sleep(0.02)
            
        samples_list = []
        if bytes_to_read > 0:
            # Read the available bytes from the serial port
            raw_data_packet = self.ser.read(bytes_to_read)

            if self.virtual:
                data_packet = np.frombuffer(raw_data_packet, dtype=np.uint16).reshape(-1, 64)
                return data_packet
            
            data_packet = np.frombuffer(raw_data_packet, dtype=np.uint8)
            number_of_packet = int(len(data_packet) / 128)

            # Process the data packet
            processed_packets = self.process_packet(data_packet, number_of_packet, validate=(not self.virtual))
            for packet in processed_packets:
                samples = np.asarray(struct.unpack(self.fmt, packet), dtype=np.int16)[
                    self.channel_map
                ]
                samples_list.append(samples)
        return np.array(samples_list)

    def write(self, data: np.ndarray, labels: Union[np.ndarray, None] = None):
        self.ser.write(data.astype(np.int16).tobytes())
        if labels is not None:
            self.ser.write(labels.astype(np.uint8).tobytes())

    def clear(self):
        """
        Clear the serial port input buffer.
        :return: None
        """
        self.ser.reset_input_buffer()

    def __del__(self):
        self.close()


class TcpStreamer(EmagerStreamerInterface):
    """
    Stream and receive EMG data over TCP.
    """

    def __init__(self, port: int, host: str = "localhost", listen: bool = False):
        """
        Create a TCP Streamer. It can read and write data over TCP.

        Params:
            - port: TCP port
            - host: Hostname
            - listen: If True, listen for incoming connections. If False, connect to a pre-existing TCP listener.
        """
        self.conn = None
        self.is_server = listen
        self.port = port
        self.host = host
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        if listen:
            self.sock.bind((host, port))
            self.sock.listen()

        self.open()

    def close(self):
        """
        Close TCP socket.
        """
        self.sock.close()

        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def open(self):
        """
        Start listening on TCP port.
        """
        if self.is_server:
            self.conn, self.addr = self.sock.accept()
            log.info(f"Connected to {self.addr}")
        else:
            self.sock.connect((self.host, self.port))
            self.conn = self.sock

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
        data = self.conn.recv(16384)
        if not data:
            return np.ndarray((0, 64))
        return np.frombuffer(data, dtype=np.int16).reshape((-1, 64))

    def write(self, data: np.ndarray, labels: Union[np.ndarray, None] = None):
        self.conn.sendall(data.astype(np.int16).tobytes())

    def __del__(self):
        self.close()


def socat_tcp_serial(tcp_port: int, serial_port: str) -> sp.Popen:
    """
    Use `socat` in a subprocess to forward data from a `tcp_port` to a `serial_port`.
    Especially useful with `SerialStreamer` and `TcpStreamer`.

    The `TcpStreamer` should be initialized with `listen=False` AFTER calling this function.

    It will open a serial port at the given `serial_port` file descriptor.

    Returns the process.
    """
    proc = sp.Popen(
        [
            "socat",
            f"TCP-LISTEN:{tcp_port}",
            f"pty,rawer,link={serial_port}",
        ]
    )
    time.sleep(3)
    return proc


def socat_serial_serial(serial_port_1: str, serial_port_2: str) -> sp.Popen:
    """
    Use `socat` in a subprocess, creating two virtual serial ports.

    Returns the process.

    Example:

    ```python
    proc = socat_serial_serial("/tmp/tty0", "/tmp/tty1")
    client_streamer = SerialStreamer("/tmp/tty0", 115200, True)
    server_streamer = SerialStreamer("/tmp/tty1", 115200, True)
    for i in range(10):
        server_streamer.write(np.random.randint(0, 1024, (10, 64)))
        print(client_streamer.read().shape)
        time.sleep(1)
    ```
    """
    proc = sp.Popen(
        [
            "socat",
            f"pty,rawer,link={serial_port_1}",
            f"pty,rawer,link={serial_port_2}",
        ]
    )
    time.sleep(3)
    return proc


if __name__ == "__main__":

    PORT = 2341
    VSERIAL = "/tmp/tty0"
    VSERIAL2 = "/tmp/tty1"
    BAUD = 230400

    proc = socat_tcp_serial(PORT, VSERIAL)
    # proc = socat_serial_serial(VSERIAL, VSERIAL2)

    server_streamer = TcpStreamer(PORT, "localhost", False)
    # server_streamer = SerialStreamer(VSERIAL2, BAUD, True)

    client_streamer = SerialStreamer(VSERIAL, BAUD, True)

    for i in range(10):
        print("-" * 20)
        data = np.random.randint(0, 1024, (10, 64))
        print("Sending data of size:", data.size)
        server_streamer.write(data)
        print("Data sent.")
        ret = client_streamer.read()
        print("Received data of size:", ret.size)

        time.sleep(0.5)

    proc.kill()
