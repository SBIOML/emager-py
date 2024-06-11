import time
from emager_py.streamers import TcpStreamer, SerialStreamer, socat_tcp_serial, socat_serial_serial
import sys
import threading
from emager_py.utils import utils
from emager_py.utils.find_usb import virtual_port

utils.set_logging()

BAUDRATE = 1500000
DATASET_PATH = "C:\GIT\Datasets\EMAGER/"

# If not on linux use VSPD to paired virtual serial ports 
# https://www.virtual-serial-port.org/
#  use port COM1 and COM2
PORT = virtual_port(DATASET_PATH, BAUDRATE)

client_streamer = SerialStreamer(PORT, BAUDRATE, True)

def read_data_thread(stop_event: threading.Event):
    while not stop_event.is_set():
        try:
            ret = client_streamer.read()
            print(f"Recived data ({ret.shape}) : \n {ret}")
            time.sleep(0.5)
        except Exception as e:
            print(f"Error reading data: {e}")
            break

stop_thread = threading.Event()
read_thread = threading.Thread(target=read_data_thread, args=(stop_thread,), daemon=True)

try:
    read_thread.start()
    print("Data reader thread started")
    while True:
        time.sleep(1)
finally:
    print("Exiting...")
    stop_thread.set()
    print("Goodbye!")
    
