import time
import numpy as np
from emager_py.streamers import TcpStreamer, SerialStreamer, socat_tcp_serial, socat_serial_serial
import sys
from emager_py.data_processings import data_generator as dg
import threading
from emager_py.utils import utils

utils.set_logging()

# Default serial port (change as needed)
PORT1 = '/dev/ttyV1' if sys.platform.startswith('linux') else 'COM12'
PORT2 = '/dev/ttyV2' if sys.platform.startswith('linux') else 'COM13'

sample_rate = 1000 # Hz
baudrate = 1500000
datasetpath = "C:\GIT\Datasets\EMAGER"

# If not on linux use VSPD to paired virtual serial ports
# https://www.virtual-serial-port.org/
if sys.platform.startswith('linux'):
    proc = socat_serial_serial(PORT1, PORT2)
    time.sleep(1)

server_streamer = SerialStreamer(PORT1, baudrate, True)
client_streamer = SerialStreamer(PORT2, baudrate, True)

data_generator = dg.EmagerDataGenerator(
    server_streamer, datasetpath, sample_rate, 50, True 
)
emg, lab = data_generator.prepare_data("000", "001")

def read_data_thread():
    while running:
        try:
            ret = client_streamer.read()
            print(f"Recived data ({ret.shape}) : \n {ret}")
            time.sleep(0.5)
        except Exception as e:
            print(f"Error reading data: {e}")
            break

running = True
read_thread = threading.Thread(target=read_data_thread, daemon=True)

try:
    read_thread.start()
    print("Data reader thread started")
    generator_thread = data_generator.start()
    print("Data generator thread started")
    while True:
        time.sleep(1)
finally:
    print("Exiting...")
    running = False
    server_streamer.close()
    client_streamer.close()
    generator_thread.join()
    read_thread.join()
    if sys.platform.startswith('linux'):
        proc.kill()
    
