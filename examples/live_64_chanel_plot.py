import emager_py.utils.utils as eutils
import emager_py.streamers as streamers
from emager_py.visualization.visualize import RealTimeOscilloscope
from emager_py.utils.find_usb import find_psoc, virtual_port

eutils.set_logging()

SAMPLING_RATE = 1000
BAUDRATE = 1500000
CHANNELS = 64
REFRESH_RATE = 30
ACCUMULATION = 3

VIRTUAL = False

if VIRTUAL:
    DATASET_PATH = "C:\GIT\Datasets\EMAGER/"
    PORT = virtual_port(DATASET_PATH, BAUDRATE)
    print("Data generator thread started")
else:
    PORT = find_psoc()

print("Starting client and oscilloscope...")
stream_client = streamers.SerialStreamer(PORT, BAUDRATE, VIRTUAL)
oscilloscope = RealTimeOscilloscope(stream_client, CHANNELS, SAMPLING_RATE, ACCUMULATION, REFRESH_RATE)
oscilloscope.run()