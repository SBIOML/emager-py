import threading
import time

import numpy as np

from emager_py.data.data_generator import EmagerDataGenerator
from emager_py.streamers import TcpStreamer
from emager_py.utils import utils

utils.set_logging()

batch = 10
dataset_path = "/Users/gabrielgagne/Documents/Datasets/EMAGER/"


def sender_task():
    global dataset_path, batch

    sender = TcpStreamer(4444, listen=True)
    dg = EmagerDataGenerator(sender, dataset_path, 1000, batch, True)
    dg.prepare_data("004", "001")
    dg.serve_data(True)
    print("Len of generated data: ", len(dg))


def main():
    # First, start the server process in a separate thread (or even on a remote machine!). It will wait for a TCP client to connect to it.
    t = threading.Thread(target=sender_task)
    t.start()
    time.sleep(3)

    # Now, start the client process. It will connect to the server and start receiving data.
    reader = TcpStreamer(4444, listen=False)
    num_received = 0
    while num_received < 10000:
        data = reader.read()
        num_received += len(data)
        if len(data) > 0:
            print(f"Received data with shape {data.shape}")
    t.join()


if __name__ == "__main__":
    main()
