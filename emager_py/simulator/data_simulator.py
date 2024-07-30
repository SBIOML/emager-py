import serial.tools.list_ports
import time
import threading
import emager_py.data.data_processing as dp
import numpy as np


class EmagerSimulator:
    def __init__(self, prepared_data:np.array, sampling_rate, port, baudrate=1500000):
        self.sampling_rate = sampling_rate
        self.streamer = serial.Serial(port, baudrate, timeout=1)
        self.stop_event = threading.Event()

        self.prepare_data(prepared_data)

    def prepare_data(self, data):
        prepared_data = np.concatenate(data, axis=0)
        prepared_data = prepared_data.reshape(-1, 64)
        data_unmap = dp.unmap(np.array(prepared_data)).astype(np.int16)
        
        # unpack 8-bit data
        data_packet_array = data_unmap.astype(np.int16).view(np.uint8)
        self.emg = data_packet_array.reshape((-1, 128)).astype(
            np.uint8
        )
        print(f"Prepared 16bit data of shape {data_unmap.shape}. i.e. data_unmap[0] \n  {data_unmap[0]}  ")
        print(f"Prepared 8bit data of shape {self.emg.shape}. i.e. emg[0] \n  {self.emg[0]}  ")
        samples = [int.from_bytes(bytes([self.emg[0][s*2], self.emg[0][s*2+1]]), 'little',signed=True) for s in range(64)]
        print(f"Samples of emg[0]: {samples}")

    def start(self):
        """
        Start serving data deamon thread.
        """
        self.thread = threading.Thread(target=self._serve_data,  daemon=True)
        self.thread.start()
    
    def stop(self):
        self.stop_event.set()
        self.thread.join()

    def _serve_data(self):
        interval = 1.0 / self.sampling_rate  # Time between data packets
        data_index = 0
        total_data = len(self.emg)
        
        while data_index < total_data and not self.stop_event.is_set():
            start_time = time.time()
            
            # Send the data packet
            data_packet = self.emg[data_index]
            self._send_packet(data_packet)
            
            # Move to the next data packet
            data_index += 1
            
            # Calculate elapsed time and sleep to maintain the sampling rate
            elapsed_time = time.time() - start_time
            sleep_time = max(0, interval - elapsed_time)  # Avoid negative sleep time
            time.sleep(sleep_time)

        print("No more data to send. Stopping the simulator.")

    def _send_packet(self, data_packet):
        rolled_packet = dp.unroll_starting_point(data_packet)
        # print(f"Sending data: {rolled_packet}")
        packet_bytes = rolled_packet.tobytes()
        # print(f"Sending bytes: {packet_bytes}")
        samples = [int.from_bytes(bytes([packet_bytes[s*2], packet_bytes[s*2+1]]), 'big',signed=True) for s in range(64)]
        # print(f"Sending samples: {samples}")
        self.streamer.write(packet_bytes)